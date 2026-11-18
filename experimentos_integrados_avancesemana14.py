"""PASO 4 — Barridos aislados, métricas auditables y salida por ejecución.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from tablas_historicas import exportar_historicos, FuenteHistorica, historicos_tabla, construir_comparacion

TOLERANCIA = 0.05
RF_CLASIFICADOR_REFERENCIA = {'n_estimators':200, 'min_samples_leaf':2,
                              'max_depth':None, 'class_weight':'balanced',
                              'random_state':42, 'n_jobs':1}
RF_REGRESOR_FIJO = {'n_estimators':200, 'min_samples_leaf':2,
                    'random_state':42, 'n_jobs':1}
COL_METRICAS = ['MAE','RMSE','R2','Pearson']
CONJUNTOS_TEST = ('Test Labels','Test Designs')


def _hash_archivo(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloque)
    return h.hexdigest()


def _csv(df, salida, nombre):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida / nombre, index=False, float_format='%.16g')
    return salida / nombre


def _clave(modelo, param):
    return hashlib.sha256(json.dumps([modelo,param], sort_keys=True).encode()).hexdigest()[:12]


def _datos(api):
    train = api.cargar_multicorner('treated_labels_train')
    conjuntos = {'Test Labels':api.cargar_multicorner('treated_labels'),
                 'Test Designs':api.cargar_multicorner('treated_test_designs')}
    columnas = [c for c in train.columns if c not in api.COLUMNAS_EXCLUIR]
    if 'Delay_Typical' not in columnas:
        raise ValueError('Delay_Typical debe ser entrada')
    X = api.preparar_X(train, columnas)
    Xt = {k:api.preparar_X(t, columnas) for k,t in conjuntos.items()}
    i_train, i_validacion = train_test_split(np.arange(len(train)),
                                    test_size=.20, random_state=api.RANDOM_STATE)
    return train, conjuntos, columnas, X, Xt, i_train, i_validacion


def _evaluacion(datos, esquina, X, tolerancia=TOLERANCIA):
    dr = datos[f'Delta_{esquina}'].to_numpy(dtype=float)
    yr = datos[f'Delay_{esquina}'].to_numpy(dtype=float)
    typ = datos['Delay_Typical'].to_numpy(dtype=float)
    mask = np.isfinite(dr) & np.isfinite(yr) & np.isfinite(typ)
    if not mask.any():
        raise ValueError(f'No hay casos válidos para {esquina}')
    X_ok = X.iloc[np.flatnonzero(mask)]
    return {'X':X_ok,'datos':datos.iloc[np.flatnonzero(mask)],
            'delta':dr[mask], 'delay':yr[mask], 'typical':typ[mask],
            'clase':(np.abs(dr[mask]) <= tolerancia).astype(int),
            'n_original':len(datos), 'n':int(mask.sum())}


def _metricas_clase(real, pred):
    # Orden fijo 0=requiere regresión; 1=Typical.
    matriz = confusion_matrix(real, pred, labels=[0, 1])
    pr, rc, f1, soporte = precision_recall_fscore_support(
        real, pred, labels=[0,1], zero_division=0)
    valores = {
        'N':len(real), 'Accuracy':float(accuracy_score(real,pred)),
        'Balanced_Accuracy':float(balanced_accuracy_score(real,pred)),
        'N_real0_pred0':int(matriz[0,0]),
        'N_real0_pred1':int(matriz[0,1]),
        'N_real1_pred0':int(matriz[1,0]),
        'N_real1_pred1':int(matriz[1,1]),
        'N_enviado_regresor':int(np.sum(pred==0)),
        'N_enviado_Typical':int(np.sum(pred==1)),
        'Recall_clase_0':float(rc[0]),
        'F1_clase_0':float(f1[0]),
    }
    por_clase=[]
    for i in (0,1):
        por_clase.append({'Clase':i,
                          'Significado':'Requiere regresor' if i==0 else 'Usar Typical',
                          'N_real':int(soporte[i]),
                          'Precision':float(pr[i]),'Recall':float(rc[i]),'F1':float(f1[i])})
    assert matriz.sum()==len(real)
    return valores,por_clase


def _metricas_sistema(api, ev, clases_pred, delta_regresor):
    clases_pred = np.asarray(clases_pred,dtype=int)
    if len(clases_pred) != ev['n'] or len(delta_regresor) != ev['n']:
        raise ValueError('Dimensiones incompatibles entre clasificador y regresor')
    delta = np.where(clases_pred == 0, delta_regresor, 0.)
    delay = ev['typical'] + delta
    a=api.metricas(ev['delay'],delay)
    b=api.metricas(ev['delta'],delta)
    d={f'{k}_Delay':a[k] for k in COL_METRICAS}
    d.update({f'{k}_Delta':b[k] for k in COL_METRICAS})
    return d,delta,delay


def _candidato(modelo, params):
    if modelo=='RandomForestClassifier':
        return RandomForestClassifier(**params)
    if modelo=='GradientBoostingClassifier':
        return GradientBoostingClassifier(**params)
    raise ValueError(f'Clasificador no contemplado: {modelo}')


def _mejor_validacion(tabla):
    # Se elige por RMSE final. Recall clase 0 desempata, no reemplaza RMSE.
    df=tabla.sort_values(['RMSE_Delay','Recall_clase_0','Balanced_Accuracy','ID'],
                         ascending=[True,False,False,True],na_position='last')
    if df.empty or not np.isfinite(df.iloc[0]['RMSE_Delay']):
        raise ValueError('No hay RMSE de validación válido para seleccionar')
    return df.iloc[0]


def _cargar_historicos(historicos):
    if not historicos:
        return pd.DataFrame()
    fuente=FuenteHistorica(historicos)
    try:
        tabla,_,_,_=historicos_tabla(fuente)
        return tabla
    finally:
        fuente.cerrar()


def _tabla_comparacion_actual(api, conjuntos, esq_modelos, hist):
    """Comparación de Delay final; métricas antiguas se marcan históricas."""
    filas=[]
    for esquina, pred_por_conjunto in esq_modelos.items():
        for nombre, (ev, pred_delta, pred_delay, nombre_modelo, parametros) in pred_por_conjunto.items():
            metr = api.metricas(ev['delay'],pred_delay)
            baseline = api.metricas(ev['delay'],ev['typical'])
            reg={'Origen':'Nueva ejecución (datasets actuales)', 'Esquina':esquina,
                 'Objetivo_entrenamiento':'Delta', 'Metrica_evaluada':'Delay reconstruido',
                 'Modelo':nombre_modelo,'Experimento':json.dumps(parametros,sort_keys=True),
                 'Conjunto':nombre,'N_evaluacion':ev['n'],'Tolerancia':TOLERANCIA}
            reg.update(metr)
            filas.append(reg)
            base={**reg,'Modelo':'Baseline Typical','Objetivo_entrenamiento':'Delay',
                  'Experimento':'Typical como prediccion'}
            base.update(baseline)
            filas.append(base)
    tabla=pd.DataFrame(filas)
    if not hist.empty:
        subset=hist.loc[hist['Conjunto'].isin(['Validacion','Test Labels','Test Designs'])].copy()
        subset['N_evaluacion']=np.nan
        subset['Tolerancia']=np.nan
        subset['Origen']=subset['Origen']+' — datos originales no cotejables por hash'
        tabla=pd.concat([tabla,subset],ignore_index=True,sort=False)
    compar=construir_comparacion(tabla)
    compar['Esquema_comparacion']=np.where(compar['Origen'].str.startswith('Nueva ejecución'),
          'Mismos CSV e índices actuales',
          'Histórico; comprobar identidad de datos y split antes de atribuir diferencias')
    return compar


def _predicciones_detalladas(esquina,nombre,ev,clases_pred,delta,delay,params,origen):
    datos=ev['datos']
    columnas=[c for c in ['Design','row_id','Previous_description','Description'] if c in datos.columns]
    out=datos[columnas].reset_index(drop=True).copy()
    out.insert(0,'Esquina',esquina)
    out.insert(1,'Conjunto',nombre)
    out['Modelo']=origen
    out['Experimento']=json.dumps(params,sort_keys=True)
    out['Delay_Typical']=ev['typical']
    out['Delta_real']=ev['delta']
    out['Delta_predicho']=delta
    out['Delay_real']=ev['delay']
    out['Delay_predicho']=delay
    out['Clase_real']=ev['clase']
    out['Clase_predicha']=clases_pred
    out['Usa_regresor']=np.asarray(clases_pred)==0
    out['Error_firmado']=delay-ev['delay']
    out['Error_absoluto']=np.abs(out['Error_firmado'].to_numpy())
    return out


def ejecutar_clasificador(api,salida,datos,historicos=None):
    train, conjuntos, columnas, X, Xt, i_train, i_validacion=datos
    print('\nPASO 4: SOLO BARRIDO DE CLASIFICADORES — tolerancia 0.05')
    all_v, all_c, all_selected, all_final, all_fclass, all_predict=[],[],[],[],[],[]
    all_extremos=[]
    all_val_pred=[]
    all_val_extremos=[]
    nuevos_por_esq={}
    cache_pars_por_esquina={}
    gestor=getattr(api,'_GESTOR_EJECUCION',None)
    for esquina in ('Slow','Fast'):
        print(f'\n-- Esquina {esquina} --')
        delta_train=train[f'Delta_{esquina}'].to_numpy(dtype=float)[i_train]
        validos_train=np.isfinite(delta_train)
        X_parcial=X.iloc[i_train].iloc[np.flatnonzero(validos_train)]
        y_parcial=delta_train[validos_train]
        y_class=(np.abs(y_parcial)<=TOLERANCIA).astype(int)
        if len(np.unique(y_class))!=2:
            raise ValueError(f'{esquina}: entrenamiento requiere ambas clases')
        imputer_clf=SimpleImputer(strategy='median')
        X_clf=imputer_clf.fit_transform(X_parcial)
        ev_v=_evaluacion(train.iloc[i_validacion],esquina,X.iloc[i_validacion])
        X_v=imputer_clf.transform(ev_v['X'])
        # Regresor de referencia sin cambios; se entrena una sola vez para
        # Validación y se predice en todos los segmentos una sola vez.
        reg=Pipeline([('imputer',SimpleImputer(strategy='median')),
                      ('modelo',RandomForestRegressor(**RF_REGRESOR_FIJO))])
        mask_reg=np.abs(y_parcial)>TOLERANCIA
        if not mask_reg.any():
            raise ValueError(f'{esquina}: no hay casos para entrenar regresor')
        clave_reg=f'clasificador|regresor_validacion|{esquina}'
        cache_reg=gestor.cargar(clave_reg) if gestor else None
        if cache_reg is None:
            if gestor: gestor.registrar(clave_reg,'INICIADO')
            reg.fit(X_parcial.iloc[np.flatnonzero(mask_reg)],y_parcial[mask_reg])
            delta_reg_v=np.asarray(reg.predict(ev_v['X']),dtype=float)
            if gestor: gestor.guardar(clave_reg,delta_reg_v,contexto={'esquina':esquina,'etapa':'validacion'})
        else:
            delta_reg_v=np.asarray(cache_reg,dtype=float)
            if len(delta_reg_v)!=ev_v['n']:
                raise RuntimeError(f'{clave_reg}: checkpoint con dimensiones incorrectas')

        cache={}
        cache_clases={}
        registros=[]
        def probar(nombre, params, etapa):
            identidad=_clave(nombre,params)
            if identidad in cache:
                return cache[identidad]
            clave_candidato=f'clasificador|validacion|{esquina}|{identidad}'
            guardado=gestor.cargar(clave_candidato) if gestor else None
            if guardado is None:
                if gestor: gestor.registrar(clave_candidato,'INICIADO',{'modelo':nombre,'parametros':params})
                model=_candidato(nombre,params)
                model.fit(X_clf,y_class)
                clase_v=np.asarray(model.predict(X_v),dtype=int)
                metr_clase,por_clase=_metricas_clase(ev_v['clase'],clase_v)
                metr_del,_,_=_metricas_sistema(api,ev_v,clase_v,delta_reg_v)
            else:
                clase_v=guardado['clases']
                metr_clase,por_clase=_metricas_clase(ev_v['clase'],clase_v)
                metr_del,_,_=_metricas_sistema(api,ev_v,clase_v,delta_reg_v)
                print(f'  REANUDADO clasificador {esquina} {identidad}')
            fila={'Esquina':esquina,'Conjunto':'Validacion','Modelo':nombre,
                  'Etapa':etapa,'ID':identidad,'Parametros':json.dumps(params,sort_keys=True),
                  'Tolerancia':TOLERANCIA, 'N_entrenamiento':len(X_parcial),
                  'N_evaluacion':ev_v['n'], 'N_entrenamiento_regresor':int(mask_reg.sum())}
            fila.update(metr_clase)
            fila.update(metr_del)
            cache[identidad]=fila
            cache_clases[identidad]=clase_v
            registros.append(fila)
            if guardado is None and gestor:
                gestor.guardar(clave_candidato,{'clases':clase_v},
                    contexto={'esquina':esquina,'modelo':nombre,'parametros':params,
                              'RMSE_Delay':metr_del['RMSE_Delay']})
            _csv(pd.DataFrame(all_v+registros),salida,'clasificador_validacion_progreso.csv')
            for clase in por_clase:
                all_c.append({'Esquina':esquina,'Conjunto':'Validacion','Modelo':nombre,
                              'ID':identidad,'Parametros':json.dumps(params,sort_keys=True),
                              **clase})
            print(f'  {etapa:12} {nombre:26} {identidad} RMSE {metr_del["RMSE_Delay"]:.6f} '
                  f'Recall[0] {metr_clase["Recall_clase_0"]:.3f}')
            return fila

        for n in (100,200,300,500):
            params=dict(RF_CLASIFICADOR_REFERENCIA,n_estimators=n)
            probar('RandomForestClassifier',params,'arboles')
        mejor1=_mejor_validacion(pd.DataFrame(registros))
        params1=json.loads(mejor1['Parametros'])
        for hoja in (1,2,4,8,16):
            params=dict(params1,min_samples_leaf=hoja)
            probar('RandomForestClassifier',params,'hojas')
        mejor2=_mejor_validacion(pd.DataFrame([x for x in registros
                            if x['Modelo']=='RandomForestClassifier']))
        params2=json.loads(mejor2['Parametros'])
        for prof in (10,20,None):
            params=dict(params2,max_depth=prof)
            probar('RandomForestClassifier',params,'profundidad')
        # Alternativa acordada con el profesor. Sin mezclar estimadores regresores.
        params_gb={'n_estimators':100,'learning_rate':0.05,'max_depth':3,
                   'min_samples_leaf':2,'random_state':42}
        probar('GradientBoostingClassifier',params_gb,'alternativa')
        ganador=_mejor_validacion(pd.DataFrame(registros))
        seleccion=dict(ganador)
        seleccion['Criterio']='RMSE_Delay de Validacion; Recall clase 0 como desempate'
        all_selected.append(seleccion)
        all_v.extend(registros)
        print(f'GANADOR {esquina}: {ganador["Modelo"]} {ganador["Parametros"]}')
        # La predicción elegida de validación se reutiliza sin reentrenar.
        _,dp_v,yp_v=_metricas_sistema(api,ev_v,cache_clases[ganador['ID']],delta_reg_v)
        pred_v=_predicciones_detalladas(esquina,'Validacion',ev_v,
                       cache_clases[ganador['ID']],dp_v,yp_v,
                       {'Clasificador':ganador['Modelo'],
                        'Parametros_clasificador':json.loads(ganador['Parametros']),
                        'Regresor':RF_REGRESOR_FIJO,'Tolerancia':TOLERANCIA},ganador['Modelo'])
        all_val_pred.append(pred_v)
        all_val_extremos.append(pred_v.nlargest(min(100,len(pred_v)),'Error_absoluto'))

        # Reentrenamiento FINAL con las filas de Train completo, SIN usar Test.
        delta_completo=train[f'Delta_{esquina}'].to_numpy(dtype=float)
        mask_total=np.isfinite(delta_completo)
        X_full=X.iloc[np.flatnonzero(mask_total)]
        delta_full=delta_completo[mask_total]
        y_full=(np.abs(delta_full)<=TOLERANCIA).astype(int)
        if len(np.unique(y_full))!=2:
            raise ValueError(f'{esquina}: Train completo carece de clase 0/1')
        clave_final=f'clasificador|final|{esquina}|{ganador["ID"]}'
        paquete_final=gestor.cargar(clave_final) if gestor else None
        if paquete_final is None:
            imputer_final=SimpleImputer(strategy='median')
            X_full_imp=imputer_final.fit_transform(X_full)
            final_clf=_candidato(ganador['Modelo'],json.loads(ganador['Parametros']))
            final_clf.fit(X_full_imp,y_full)
            reg_final=Pipeline([('imputer',SimpleImputer(strategy='median')),
                                ('modelo',RandomForestRegressor(**RF_REGRESOR_FIJO))])
            mask_reg_full=np.abs(delta_full)>TOLERANCIA
            reg_final.fit(X_full.iloc[np.flatnonzero(mask_reg_full)],delta_full[mask_reg_full])
            pred_finales={}
            for conjunto in CONJUNTOS_TEST:
                ev_aux=_evaluacion(conjuntos[conjunto],esquina,Xt[conjunto])
                pred_finales[conjunto]={
                    'clases':np.asarray(final_clf.predict(imputer_final.transform(ev_aux['X'])),dtype=int),
                    'delta_reg':np.asarray(reg_final.predict(ev_aux['X']),dtype=float),
                }
            paquete_final={'predicciones':pred_finales,'n_reg':int(mask_reg_full.sum())}
            if gestor: gestor.guardar(clave_final,paquete_final,
                contexto={'esquina':esquina,'clasificador':ganador['ID'],'fase':'Train completo + Test'})
        mask_reg_full=np.abs(delta_full)>TOLERANCIA
        nuevos_por_esq[esquina]={}
        params_seleccionados={'Clasificador':ganador['Modelo'],
                              'Parametros_clasificador':json.loads(ganador['Parametros']),
                              'Regresor':RF_REGRESOR_FIJO,'Tolerancia':TOLERANCIA}
        for conjunto in CONJUNTOS_TEST:
            ev=_evaluacion(conjuntos[conjunto],esquina,Xt[conjunto])
            cl_pred=np.asarray(paquete_final['predicciones'][conjunto]['clases'],dtype=int)
            dp_reg=np.asarray(paquete_final['predicciones'][conjunto]['delta_reg'],dtype=float)
            if len(cl_pred)!=ev['n'] or len(dp_reg)!=ev['n']:
                raise RuntimeError('Checkpoint final con dimensiones incompatibles')
            metr_cl,por_clase=_metricas_clase(ev['clase'],cl_pred)
            metr_sis,dp,delay=_metricas_sistema(api,ev,cl_pred,dp_reg)
            d={'Esquina':esquina,'Conjunto':conjunto,'Modelo':ganador['Modelo'],
               'ID':ganador['ID'],'Parametros':ganador['Parametros'],
               'N_entrenamiento':len(X_full), 'N_evaluacion':ev['n'],
               'N_entrenamiento_regresor':int(mask_reg_full.sum()),
               'Tolerancia':TOLERANCIA,**metr_cl,**metr_sis}
            all_final.append(d)
            for clase in por_clase:
                all_fclass.append({'Esquina':esquina,'Conjunto':conjunto,
                    'ID':ganador['ID'],'Modelo':ganador['Modelo'],
                    'Parametros':ganador['Parametros'],**clase})
            pred=_predicciones_detalladas(esquina,conjunto,ev,cl_pred,dp,delay,
                    params_seleccionados,ganador['Modelo'])
            all_predict.append(pred)
            all_extremos.append(pred.nlargest(min(100,len(pred)),'Error_absoluto'))
            nuevos_por_esq[esquina][conjunto]=(ev,dp,delay,ganador['Modelo'],params_seleccionados)
            print(f'  FINAL {esquina}/{conjunto}: RMSE {metr_sis["RMSE_Delay"]:.6f}')

        # También guardar métricas de la referencia de validación ya ejecutada.
        ref_id=_clave('RandomForestClassifier',RF_CLASIFICADOR_REFERENCIA)
        if ref_id not in cache:
            raise AssertionError('Se perdió la configuración de referencia')
        cache_pars_por_esquina[esquina]=ref_id

    resultados_v=pd.DataFrame(all_v)
    por_clase=pd.DataFrame(all_c)
    elegidos=pd.DataFrame(all_selected)
    finales=pd.DataFrame(all_final)
    por_clase_final=pd.DataFrame(all_fclass)
    for nombre,df in [
        ('clasificador_validacion_experimentos.csv',resultados_v),
        ('clasificador_validacion_por_clase.csv',por_clase),
        ('clasificador_delay_reconstruido_validacion.csv',
           resultados_v[['Esquina','Conjunto','Modelo','Etapa','ID','Parametros',
                          'N_evaluacion','MAE_Delay','RMSE_Delay','R2_Delay','Pearson_Delay',
                          'MAE_Delta','RMSE_Delta','R2_Delta','Pearson_Delta']]),
        ('clasificador_configuraciones_seleccionadas.csv',elegidos),
        ('clasificador_evaluacion_final_test.csv',finales),
        ('clasificador_evaluacion_final_por_clase.csv',por_clase_final),
        ('clasificador_predicciones_finales.csv',pd.concat(all_predict,ignore_index=True)),
        ('clasificador_errores_extremos.csv',pd.concat(all_extremos,ignore_index=True)),
        ('clasificador_predicciones_validacion_ganadores.csv',pd.concat(all_val_pred,ignore_index=True)),
        ('clasificador_errores_extremos_validacion.csv',pd.concat(all_val_extremos,ignore_index=True)),
    ]:
        _csv(df,salida,nombre)

    hist=_cargar_historicos(historicos)
    compar=_tabla_comparacion_actual(api,conjuntos,nuevos_por_esq,hist)
    _csv(compar,salida,'clasificador_comparacion_modelos.csv')
    # Tablas integradas: cada fila identifica fuente y métrica Delay reconstruido.
    _csv(compar.loc[compar['Metrica_evaluada']=='Delay reconstruido'].copy(),
         salida,'tabla_4_conjuntos_integrada.csv')
    _csv(compar.loc[compar['Metrica_evaluada']=='Delay reconstruido'].copy(),
         salida,'resumen_para_reunion_integrado.csv')
    assert not resultados_v.duplicated(['Esquina','ID']).any()
    assert len(finales)==4, 'Se esperaban 2 esquinas x 2 conjuntos Test'
    return {'configuraciones_validadas':len(resultados_v),'seleccionados':len(elegidos),
            'filas_test_final':len(finales),'ids_referencia':cache_pars_por_esquina}


def ejecutar_hibrido(api,salida,datos,historicos=None):
    train,conjuntos,columnas,X,Xt,i_train,i_validacion=datos
    print('\nPASO 4: SOLO BARRIDO DEL HIBRIDO FAST-DELTA')
    imp_v=SimpleImputer(strategy='median')
    X_tr=pd.DataFrame(imp_v.fit_transform(X.iloc[i_train]),
                     columns=X.columns,index=X.index[i_train])
    X_v=pd.DataFrame(imp_v.transform(X.iloc[i_validacion]),
                    columns=X.columns,index=X.index[i_validacion])
    imp_f=SimpleImputer(strategy='median')
    X_f=pd.DataFrame(imp_f.fit_transform(X),columns=X.columns,index=X.index)
    y_tr=train['Delta_Fast'].iloc[i_train]
    y_f=train['Delta_Fast']
    evs={'Validacion':_evaluacion(train.iloc[i_validacion],'Fast',X.iloc[i_validacion])}
    for nom in CONJUNTOS_TEST:
        evs[nom]=_evaluacion(conjuntos[nom],'Fast',Xt[nom])
    todas, todas_pred, todos_ext=[],[],[]
    cache_modelos=[]
    gestor=getattr(api,'_GESTOR_EJECUCION',None)
    for hojas in api.HYBRID_MIN_SAMPLES_LEAF_EXPERIMENTS:
        clave_hoja=f'hibrido|Fast|Delta|leaf={int(hojas)}'
        datos_hoja=gestor.cargar(clave_hoja) if gestor else None
        if datos_hoja is not None:
            print(f'  REANUDADO hibrido Fast–Delta hoja={hojas}')
            todas.extend(datos_hoja['metricas'])
            todas_pred.extend(datos_hoja['predicciones'])
            todos_ext.extend(datos_hoja['extremos'])
            cache_modelos.append(tuple(datos_hoja['estado']))
            continue
        indice_m, indice_p, indice_e=len(todas),len(todas_pred),len(todos_ext)
        if gestor: gestor.registrar(clave_hoja,'INICIADO',{'hojas':hojas})
        modv=api.HybridModel(min_samples_leaf=hojas)
        modv.fit(X_tr,y_tr,LR_type=0,custom_params=None)
        modf=api.HybridModel(min_samples_leaf=hojas)
        modf.fit(X_f,y_f,LR_type=0,custom_params=None)
        parametros={'min_samples_leaf':int(hojas),'max_depth':9,'max_features':15,
                    'LR_type':0,'tree_random_state':10}
        ident=_clave('Hybrid Model',parametros)
        # Las hojas reales se calculan de la estructura entrenada, no se suponen.
        ids_v=modv.decision_tree_object.apply(X_tr)
        ids_f=modf.decision_tree_object.apply(X_f)
        minimo_v=int(np.unique(ids_v,return_counts=True)[1].min())
        minimo_f=int(np.unique(ids_f,return_counts=True)[1].min())
        for conjunto,ev in evs.items():
            if conjunto=='Validacion':
                X_imp=X_v.iloc[np.flatnonzero(np.isfinite(train['Delta_Fast'].iloc[i_validacion].to_numpy(dtype=float)) &
                              np.isfinite(train['Delay_Fast'].iloc[i_validacion].to_numpy(dtype=float)) &
                              np.isfinite(train['Delay_Typical'].iloc[i_validacion].to_numpy(dtype=float)))]
                modelo=modv; n_train=len(X_tr); minimo=minimo_v
            else:
                X_imp=pd.DataFrame(imp_f.transform(ev['X']),columns=X.columns,index=ev['X'].index)
                modelo=modf; n_train=len(X_f); minimo=minimo_f
            dp=api.predecir_hibrido_por_hojas(modelo,X_imp)
            yp=ev['typical']+dp
            met_d=api.metricas(ev['delta'],dp)
            met_y=api.metricas(ev['delay'],yp)
            error=np.abs(yp-ev['delay'])
            base=api.metricas(ev['delay'],ev['typical'])
            fila={'Modelo':'Hybrid Model','Esquina':'Fast','Objetivo_entrenamiento':'Delta',
                  'Conjunto':conjunto,'ID':ident,'Parametros':json.dumps(parametros,sort_keys=True),
                  'min_samples_leaf':int(hojas),'N_entrenamiento':n_train,
                  'N_evaluacion':ev['n'],'Minimo_real_hoja_train':minimo,
                  'N_hojas_modelo':len(modelo.LR_results),
                  'RMSE_Baseline_Typical':base['RMSE'],
                  'Mejora_RMSE_vs_baseline_pct':100*(base['RMSE']-met_y['RMSE'])/base['RMSE'],
                  'Error_absoluto_P95':float(np.percentile(error,95)),
                  'Error_absoluto_P99':float(np.percentile(error,99)),
                  'Error_absoluto_maximo':float(error.max()),
                  'N_error_abs_mayor_0_1':int((error>.1).sum()),
                  'N_error_abs_mayor_1':int((error>1).sum())}
            fila.update({f'{k}_Delta':met_d[k] for k in COL_METRICAS})
            fila.update({f'{k}_Delay':met_y[k] for k in COL_METRICAS})
            todas.append(fila)
            pred=_predicciones_detalladas('Fast',conjunto,ev,
                          np.where(np.abs(dp)>TOLERANCIA,0,1).astype(int),dp,yp,
                          parametros,'Hybrid Model')
            # El modelo híbrido NO es un clasificador; quitar etiquetas engañosas.
            pred=pred.drop(columns=['Clase_real','Clase_predicha','Usa_regresor'])
            pred['min_samples_leaf']=hojas
            todas_pred.append(pred)
            todos_ext.append(pred.nlargest(min(100,len(pred)),'Error_absoluto'))
            print(f'  hoja={hojas:2} {conjunto:12} RMSE={met_y["RMSE"]:.6f} '
                  f'mínimo_real={minimo}')
        cache_modelos.append((hojas,minimo_v,minimo_f))
        if gestor:
            gestor.guardar(clave_hoja,{
                'metricas':todas[indice_m:],
                'predicciones':todas_pred[indice_p:],
                'extremos':todos_ext[indice_e:],
                'estado':(hojas,minimo_v,minimo_f),
            },contexto={'hojas':hojas,'conjuntos':list(evs)})
        _csv(pd.DataFrame(todas),salida,'hibrido_fast_delta_progreso.csv')
    resumen=pd.DataFrame(todas)
    ranking=resumen.loc[resumen.Conjunto=='Validacion'].sort_values(
        ['RMSE_Delay','min_samples_leaf'],ascending=[True,True])
    mejor=int(ranking.iloc[0]['min_samples_leaf'])
    seleccionado=resumen.loc[resumen.min_samples_leaf==mejor].copy()
    seleccionado['Criterio']='RMSE_Delay solamente en Validacion'
    _csv(resumen,salida,'hibrido_fast_delta_todas_configuraciones.csv')
    _csv(seleccionado,salida,'hibrido_fast_delta_seleccion_validacion.csv')
    _csv(pd.concat(todas_pred,ignore_index=True),salida,'hibrido_fast_delta_predicciones_individuales.csv')
    _csv(pd.concat(todos_ext,ignore_index=True),salida,'hibrido_fast_delta_errores_extremos.csv')
    # La hoja 4 original se conserva; el ganador se identifica explícitamente.
    compar=[]
    for conjunto,ev in evs.items():
        b=api.metricas(ev['delay'],ev['typical'])
        compar.append({'Esquina':'Fast','Conjunto':conjunto,'Modelo':'Baseline Typical',
                       'Origen':'Nueva ejecución','Parametros':'{}',
                       **{k:b[k] for k in COL_METRICAS}})
    for _,r in resumen.iterrows():
        compar.append({'Esquina':'Fast','Conjunto':r['Conjunto'],'Modelo':
                       ('Hybrid Model (seleccionado en Validacion)' if int(r['min_samples_leaf'])==mejor
                        else 'Hybrid Model (candidato)'),
                       'Origen':'Nueva ejecución', 'Parametros':r['Parametros'],
                       **{k:r[f'{k}_Delay'] for k in COL_METRICAS}})
    if historicos:
        hist=_cargar_historicos(historicos)
        for _,r in hist.loc[(hist.Esquina=='Fast') &
                             (hist['Conjunto'].isin(['Validacion','Test Labels','Test Designs'])) &
                             (hist['Modelo'].str.contains('Random Forest|Hybrid Model',na=False))].iterrows():
            compar.append({'Esquina':'Fast','Conjunto':r['Conjunto'],'Modelo':r['Modelo'],
                           'Origen':r['Origen']+' (identidad CSV no comprobada)',
                           'Parametros':r['Experimento'],**{k:r[k] for k in COL_METRICAS}})
    tabla_hibrido=pd.DataFrame(compar)
    _csv(tabla_hibrido,salida,'hibrido_fast_delta_comparacion.csv')
    _csv(seleccionado,salida,'resumen_para_reunion_hibrido.csv')
    # Vista general sin borrar la procedencia: histórico y barrido nuevo.
    if historicos:
        h=_cargar_historicos(historicos)
        h=h.rename(columns={'Experimento':'Parametros'})
        h['Metrica_evaluada']='Delay reconstruido'
        tabla_hibrido['Metrica_evaluada']='Delay reconstruido'
        tabla_hibrido=pd.concat([tabla_hibrido,h],ignore_index=True,sort=False)
    _csv(tabla_hibrido,salida,'tabla_4_conjuntos_integrada.csv')
    _csv(tabla_hibrido,salida,'resumen_para_reunion_integrado.csv')
    assert len(resumen)==len(api.HYBRID_MIN_SAMPLES_LEAF_EXPERIMENTS)*3
    return {'configuraciones':len(cache_modelos),'filas_metricas':len(resumen),
            'mejor_hoja':mejor,'minimos_reales_por_modelo':cache_modelos}


def _manifest(api, datos, salida, modo, historicos=None):
    _train, _tests, columnas, _X, _Xt, itrain, ival=datos
    raiz=Path(api.BASE_DIR)
    entradas = {}
    for prefijo in ('treated_labels_train','treated_labels','treated_test_designs'):
        for c in ('typical','slow','fast'):
            ruta=raiz/f'{prefijo}_{c}.csv'
            entradas[ruta.name]=_hash_archivo(ruta) if ruta.exists() else None
    d={'Modo':modo,'Generado':datetime.now().isoformat(timespec='seconds'),
       'Ruta_codigo':str(Path(api.__file__).resolve()),
       'SHA256_codigo':_hash_archivo(Path(api.__file__).resolve()),
       'SHA256_datos':entradas,
       'Version_sklearn':__import__('sklearn').__version__,
       'Random_state':api.RANDOM_STATE,
       'N_train_parcial':len(itrain),'N_validacion':len(ival),
       'SHA256_indices_train':hashlib.sha256(np.asarray(itrain,dtype='int64').tobytes()).hexdigest(),
       'SHA256_indices_validacion':hashlib.sha256(np.asarray(ival,dtype='int64').tobytes()).hexdigest(),
       'Features':columnas,'Tolerancia':TOLERANCIA,
       'Fuente_historica':str(historicos) if historicos else None,
       'Nota_historico':'No se dispone de SHA256 de datos en los CSV históricos; comparabilidad no demostrada'}
    (Path(salida)/'manifiesto_ejecucion.json').write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')



def ejecutar_cli(api,argv=None):
    """Ejecuta una corrida independiente; solo --reanudar permite reutilizar checkpoints."""
    from contextlib import ExitStack
    from joblib import parallel_config
    from ejecucion_segura import GestorEjecucion, crear_identidad, sha256_archivo
    from tablas_actuales import generar_tablas
    import traceback
    import os
    import warnings
    parser=argparse.ArgumentParser(description='TFG multicorner, ejecución portable y reanudable')
    parser.add_argument('--modo',choices=['clasificador','hibrido','tablas','completo','todo'],default='todo',
                        help='Por defecto todo: equivale a ejecutar el programa principal sin argumentos')
    parser.add_argument('--datos-dir',default=None,help='Carpeta con los nueve CSV multicorner (por defecto, junto al script)')
    parser.add_argument('--salida-raiz',default=None,help='Carpeta base para guardar corridas nuevas')
    parser.add_argument('--reanudar',default=None,help='Ruta exacta de corrida anterior con checkpoints')
    parser.add_argument('--desde-corrida',default=None,help='Solo modo tablas: regenerar tablas desde una corrida PROPIA terminada y verificada')
    parser.add_argument('--n-jobs',type=int,default=1,help='Trabajadores RF/XGBoost (1: seguro en Windows; 2/4: hilos)')
    parser.add_argument('--backend',choices=['threading','loky'],default='threading',
                        help='Backend joblib; threading evita comunicación multiproceso (recomendado)')
    parser.add_argument('--historico-zip',default=None,help='OPCIONAL: ZIP/carpeta antigua, exportada aparte, nunca usada como resultados actuales')
    args=parser.parse_args(argv)
    if args.n_jobs==0 or args.n_jobs < -1: parser.error('--n-jobs debe ser -1 o un entero positivo')
    if args.backend=='loky' and args.n_jobs!=1:
        print('ADVERTENCIA: loky usa procesos; la solución conservadora para Windows es threading o n-jobs=1.')
    if args.modo in ('todo','completo') and api.XGBRegressor is None:
        parser.error('El flujo completo requiere XGBoost. Instale: python -m pip install -r requirements.txt')
    lista=api.crear_experimentos()
    claves=[(nombre,json.dumps(params,sort_keys=True)) for nombre,params in lista]
    if len(lista)!=49 or len(set(claves))!=49:
        raise RuntimeError(f'Esperadas 49 configuraciones únicas, hay {len(lista)}; únicas={len(set(claves))}')
    # Ninguna carpeta histórica se busca de forma automática.
    api.BASE_DIR=Path(args.datos_dir).resolve() if args.datos_dir else Path(api.__file__).resolve().parent
    if args.desde_corrida:
        if args.modo != 'tablas' or args.reanudar or args.historico_zip:
            parser.error('--desde-corrida solamente admite --modo tablas y no se combina con --reanudar o históricos')
        fuente=Path(args.desde_corrida).resolve()
        meta=fuente/'control_ejecucion'/'manifiesto.json'
        fin=fuente/'control_ejecucion'/'ejecucion_completa.json'
        if not meta.is_file() or not fin.is_file():
            raise RuntimeError('La corrida indicada no está completa y verificada; no se reconstruyen tablas incompletas.')
        origen=json.loads(meta.read_text(encoding='utf-8'))
        datos=_datos(api)
        actual=crear_identidad(api,Path(api.__file__).resolve().parent,datos,
               n_jobs=origen['parametros']['n_jobs'],
               backend=origen['parametros']['backend'],modo=origen['modo'])
        if actual['fingerprint']!=origen['fingerprint']:
            raise RuntimeError('Los nueve CSV, el código, las versiones, el split o los parámetros no coinciden con la corrida.')
        if origen['modo'] not in ('todo','completo','clasificador','hibrido'):
            raise RuntimeError('Modo fuente no válido para reconstruir tablas comparativas.')
        resultado=generar_tablas(api,datos,fuente,modo=origen['modo'],
                         permitir_parcial=origen['modo'] in ('clasificador','hibrido'))
        print('Tablas regeneradas desde ESTA corrida verificada, sin entrenar: '+str(fuente))
        print('Completitud:',resultado)
        return
    if args.reanudar:
        ruta=Path(args.reanudar).resolve()
        if not ruta.is_dir(): raise FileNotFoundError(f'No existe carpeta --reanudar {ruta}')
    else:
        root=(Path(args.salida_raiz).resolve() if args.salida_raiz else
              Path(api.BASE_DIR)/'resultados_ejecuciones')
        root.mkdir(parents=True,exist_ok=True)
        from uuid import uuid4
        ruta=root/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+args.modo+'_'+uuid4().hex[:8])
        ruta.mkdir(parents=True,exist_ok=False)
    # Es posible trabajar en --modo tablas sin ningún resultado previo.
    print(f'\nModo: {args.modo}\nDatos: {api.BASE_DIR}\nCorrida: {ruta}\nReanudar: {bool(args.reanudar)}')
    api.N_JOBS=args.n_jobs
    global RF_CLASIFICADOR_REFERENCIA, RF_REGRESOR_FIJO
    RF_CLASIFICADOR_REFERENCIA=dict(RF_CLASIFICADOR_REFERENCIA,n_jobs=args.n_jobs)
    RF_REGRESOR_FIJO=dict(RF_REGRESOR_FIJO,n_jobs=args.n_jobs)
    datos=_datos(api)
    identidad=crear_identidad(api,Path(api.__file__).resolve().parent,datos,
                               n_jobs=args.n_jobs,backend=args.backend,modo=args.modo)
    gestor=GestorEjecucion(ruta,identidad,reanudar=bool(args.reanudar))
    api._GESTOR_EJECUCION=gestor
    api.RESULTS_DIR=ruta/'flujo_original'
    api.GRAFICAS_DIR=api.RESULTS_DIR/'graficas'
    api.IMPORTANCIA_DIR=api.RESULTS_DIR/'importancia'
    # El modo aislado no necesita crear los directorios del flujo principal.
    if args.modo in ('todo','completo'):
        for p in (api.RESULTS_DIR,api.GRAFICAS_DIR,api.IMPORTANCIA_DIR):
            p.mkdir(parents=True,exist_ok=True)
    def fase(etiqueta,funcion,archivos):
        clave='fase|'+etiqueta
        previo=gestor.cargar(clave)
        if previo is not None:
            for nombre,sha in previo['archivos'].items():
                if not Path(nombre).is_file() or sha256_archivo(nombre)!=sha:
                    raise RuntimeError('Archivo de fase completada falta o cambió: '+nombre)
            print('FASE REANUDADA (sin entrenar): '+etiqueta)
            return previo.get('resultado',{})
        gestor.registrar(clave,'INICIADO')
        resultado=funcion()
        faltan=[str(p) for p in archivos if not Path(p).is_file()]
        if faltan: raise RuntimeError(f'Fase {etiqueta} incompleta: faltan {faltan}')
        hashes={str(p):sha256_archivo(p) for p in archivos}
        gestor.guardar(clave,{'archivos':hashes,'resultado':resultado},contexto={'modo':args.modo})
        return resultado
    try:
        # Usar threads o ejecución secuencial para evitar comunicaciones
        # de procesos joblib que intervinieron en la traza informada.
        with parallel_config(backend=args.backend,n_jobs=args.n_jobs):
            if args.modo in ('completo','todo'):
                # En modo todo, cinco hojas se harán en el módulo nuevo,
                # sin repetirlas dentro del bloque heredado del flujo original.
                api._EJECUTAR_HIBRIDO_LEGADO=(args.modo=='completo')
                requeridos=[api.RESULTS_DIR/'metricas_objetivo3.csv',
                    api.RESULTS_DIR/'mejores_modelos.csv',
                    api.RESULTS_DIR/'resultados_clasificador_regresor_delta.csv',
                    api.RESULTS_DIR/'comparacion_hybrid_rf_baseline_fast_delta.csv',
                    api.RESULTS_DIR/'tabla_4_conjuntos.csv',
                    api.RESULTS_DIR/'resumen_para_reunion.csv']
                requeridos += [api.RESULTS_DIR/f'ranking_{c}_{o}.csv'
                                for c in ('slow','fast') for o in ('delay','delta')]
                requeridos += [api.RESULTS_DIR/f for f in (
                    'metricas_progreso.csv', 'experimentos_random_forest.csv',
                    'experimentos_validacion.csv', 'features_utilizadas.csv',
                    'diagnostico_distribucion_delta.csv', 'importancia_variables.csv',
                    'calibracion_baseline_aware.csv', 'comparacion_delta_baseline.csv',
                    'comparacion_delta_vs_delay.csv',
                    'comparacion_min_samples_leaf_fast_delta.csv',
                    'resumen_min_samples_leaf_fast_delta.csv',
                    'evaluacion_completa_fast_delta.csv',
                    'predicciones_test_labels.csv', 'predicciones_test_designs.csv',
                    'predicciones_clasificador_regresor_delta.csv',
                    'metricas_clasificador_delta_cero.csv',
                    'comparacion_clasificador_regresor_baseline_modelo_anterior.csv',
                    'modelo_hibrido_fast_delta.joblib',
                )]
                requeridos += [api.RESULTS_DIR/f'mejor_modelo_{c}_{o}.joblib'
                                for c in ('slow','fast') for o in ('delay','delta')]
                requeridos += [api.IMPORTANCIA_DIR/f'feature_importance_rf_mejor_{c}_{o}.{ext}'
                                for c in ('slow','fast') for o in ('delay','delta')
                                for ext in ('csv','png')]
                requeridos += [api.GRAFICAS_DIR/f'{conjunto}_typical_vs_{corner}.png'
                                for conjunto in ('train','validacion','test_labels','test_designs')
                                for corner in ('slow','fast')]
                requeridos += [api.GRAFICAS_DIR/f'{conjunto}_{corner}_{objetivo}.png'
                                for conjunto in ('test_labels','test_designs')
                                for corner in ('slow','fast') for objetivo in ('delay','delta')]
                if args.modo=='completo':
                    requeridos += [api.RESULTS_DIR/'experimentos_hibrido_min_samples_leaf.csv']
                fase('flujo_completo',lambda:api.main() or {'estado':'Terminado'},requeridos)
            if args.modo in ('todo','hibrido'):
                salida=ruta/'hibrido'
                fase('hibrido_cinco_hojas',lambda:ejecutar_hibrido(api,salida,datos,None),[
                    salida/'hibrido_fast_delta_todas_configuraciones.csv',
                    salida/'hibrido_fast_delta_seleccion_validacion.csv',
                    salida/'hibrido_fast_delta_predicciones_individuales.csv',
                    salida/'hibrido_fast_delta_errores_extremos.csv',
                    salida/'hibrido_fast_delta_comparacion.csv',
                    salida/'hibrido_fast_delta_progreso.csv',
                    salida/'resumen_para_reunion_hibrido.csv'])
            if args.modo in ('todo','clasificador'):
                salida=ruta/'clasificador'
                fase('clasificador_nuevo',lambda:ejecutar_clasificador(api,salida,datos,None),[
                    salida/'clasificador_validacion_experimentos.csv',
                    salida/'clasificador_configuraciones_seleccionadas.csv',
                    salida/'clasificador_evaluacion_final_test.csv',
                    salida/'clasificador_evaluacion_final_por_clase.csv',
                    salida/'clasificador_predicciones_finales.csv',
                    salida/'clasificador_comparacion_modelos.csv',
                    salida/'clasificador_validacion_progreso.csv',
                    salida/'clasificador_delay_reconstruido_validacion.csv',
                    salida/'clasificador_validacion_por_clase.csv',
                    salida/'clasificador_predicciones_validacion_ganadores.csv',
                    salida/'clasificador_errores_extremos.csv',
                    salida/'clasificador_errores_extremos_validacion.csv'])
            # Comparación SIEMPRE desde esta ejecución, jamás leer ZIP ni carpeta
            # antigua para seleccionar o completar campos numéricos actuales.
            r=generar_tablas(api,datos,ruta,modo=args.modo,
                            permitir_parcial=args.modo in ('tablas','clasificador','hibrido'))
            print('Tablas actuales:',r)
            if args.historico_zip:
                p=Path(args.historico_zip)
                if not p.exists(): raise FileNotFoundError(f'--historico-zip inexistente: {p}')
                info=exportar_historicos(str(p),ruta/'historicos_opcionales_NO_MEZCLADOS')
                print('Históricos separados sin mezclarse con la corrida actual:',info)
            # Comprobar no duplicados y número de candidatos, y que los resultados
            # principales provengan de la corrida. Fallar si incompleto.
            if args.modo in ('todo','completo'):
                general=pd.read_csv(api.RESULTS_DIR/'metricas_objetivo3.csv')
                valid=general.loc[(general.Conjunto=='Validacion') &
                                   (general.Objetivo.isin(['Delay','Delta'])) &
                                   (general.Metrica_evaluada=='Delay reconstruido')]
                for cor in ('Slow','Fast'):
                    for tar in ('Delay','Delta'):
                        v=valid[(valid.Esquina==cor)&(valid.Objetivo==tar)]
                        if len(v)!=49 or v.duplicated(['Modelo','Experimento']).any():
                            raise RuntimeError(f'Barrido incompleto o repetido: {cor} {tar}: {len(v)}')
                if len(pd.read_csv(api.RESULTS_DIR/'mejores_modelos.csv'))!=4:
                    raise RuntimeError('Deben seleccionarse 4 modelos generales')
            if args.modo in ('todo','clasificador'):
                c=pd.read_csv(ruta/'clasificador'/'clasificador_validacion_experimentos.csv')
                t=pd.read_csv(ruta/'clasificador'/'clasificador_evaluacion_final_test.csv')
                if len(c)!=22 or c.duplicated(['Esquina','ID']).any() or len(t)!=4:
                    raise RuntimeError('Clasificadores incompletos o duplicados')
            if args.modo in ('todo','hibrido'):
                h=pd.read_csv(ruta/'hibrido'/'hibrido_fast_delta_todas_configuraciones.csv')
                if len(h)!=15 or h.duplicated(['min_samples_leaf','Conjunto']).any():
                    raise RuntimeError('Barrido híbrido incompleto')
            if args.modo in ('todo','completo') and not r['completo']:
                raise RuntimeError('Faltan métricas en tablas actuales: '+str(r['pendientes']))
            if args.modo=='tablas' and not r['completo']:
                gestor.registrar('CORRIDA','TABLAS_PARCIALES',{'pendientes':r['pendientes']})
                print('\nTABLAS PARCIALES: no hay modelos entrenados en esta corrida. '
                      'No se marca como ejecución completa. Carpeta: '+str(ruta))
            else:
                gestor.terminar()
                print('\nEJECUCIÓN TERMINADA Y VERIFICADA. Carpeta: '+str(ruta))
    except BaseException as exc:
        gestor.fallo('CORRIDA',exc)
        print('\nERROR: ejecución NO terminada. Datos guardados, puede usar --reanudar.\n'+
              ''.join(traceback.format_exception_only(type(exc),exc)).strip(),file=sys.stderr)
        raise
