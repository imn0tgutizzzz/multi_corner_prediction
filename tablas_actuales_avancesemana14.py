"""Tablas de una CORRIDA ACTUAL, sin referencia a directorios históricos.

Nunca inventa métricas: todos los valores provienen de CSV generados en la
corrida actual o directamente de los nueve CSV de entrada (solo baseline).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

METRICAS=('MAE','RMSE','R2','Pearson')
CORNERS=('Slow','Fast')
SPLITS=('Train','Validacion','Test Labels','Test Designs')


def _leer(ruta):
    p=Path(ruta)
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


def _registro(fuente, modelo, params, esquina, objetivo, conjunto, met,
              *, n=None, estado='Medido', clasificacion=None):
    return {'Origen':'Ejecucion actual: '+fuente,'Modelo':str(modelo),
            'Parametros':str(params),'Esquina':esquina,'Objetivo_entrenamiento':objetivo,
            'Conjunto':conjunto,'Metrica_evaluada':'Delay reconstruido',
            'N_evaluacion':n,'Estado':estado,
            'Seleccion':clasificacion or '',
            **{k:float(met[k]) if k in met and pd.notna(met[k]) else np.nan for k in METRICAS}}


def generar_tablas(api,datos,ruta_corrida,*,modo='todo',permitir_parcial=False):
    ruta=Path(ruta_corrida)
    salida=ruta/'tablas_actuales'; salida.mkdir(parents=True,exist_ok=True)
    path_general=ruta/'flujo_original'
    metricas=_leer(path_general/'metricas_objetivo3.csv')
    mejores=_leer(path_general/'mejores_modelos.csv')
    original_clf=_leer(path_general/'resultados_clasificador_regresor_delta.csv')
    hybrid_ref=_leer(path_general/'comparacion_hybrid_rf_baseline_fast_delta.csv')
    hoja_original=_leer(path_general/'experimentos_hibrido_min_samples_leaf.csv')
    hoja_nueva=_leer(ruta/'hibrido'/'hibrido_fast_delta_todas_configuraciones.csv')
    cls_val=_leer(ruta/'clasificador'/'clasificador_validacion_experimentos.csv')
    cls_sel=_leer(ruta/'clasificador'/'clasificador_configuraciones_seleccionadas.csv')
    cls_test=_leer(ruta/'clasificador'/'clasificador_evaluacion_final_test.csv')
    train, tests, features, X, Xt, itrain, ival=datos
    tabla=[]; pendientes=[]
    # Baseline SIEMPRE calculado en esta computadora desde los nueve CSV.
    conjuntos={'Train':train.iloc[itrain], 'Validacion':train.iloc[ival],**tests}
    for corner in CORNERS:
        for split,d in conjuntos.items():
            real=d[f'Delay_{corner}'].to_numpy(dtype=float)
            typ=d['Delay_Typical'].to_numpy(dtype=float)
            m=np.isfinite(real)&np.isfinite(typ)
            tabla.append(_registro('CSV multicorner', 'Baseline Typical','Typical como prediccion',
                   corner,'Delay',split,api.metricas(real[m],typ[m]),n=int(m.sum()),clasificacion='Baseline'))
    if metricas.empty:
        pendientes.append('No está la tabla del barrido general; solo baseline disponible.')
    else:
        # Todos los entrenamientos quedan en metricas_objetivo3.csv;
        # añadir un resumen compacto con el ganador de cada objetivo y split.
        if mejores.empty:
            pendientes.append('No está mejores_modelos.csv del flujo completo.')
        else:
            if len(mejores)!=4 or mejores.duplicated(['Esquina','Objetivo']).any():
                raise RuntimeError('mejores_modelos.csv incompleto o repetido')
            for _,win in mejores.iterrows():
                corner=win['Esquina']; target=win['Objetivo']; model=win['Modelo']; params=win['Experimento']
                for split in ('Train final','Validacion','Test Labels','Test Designs'):
                    obj=f'{target} (final)' if split!='Validacion' else target
                    q=metricas[(metricas['Modelo']==model)&(metricas['Esquina']==corner)&
                         (metricas['Objetivo']==obj)&(metricas['Experimento']==params)&
                         (metricas['Conjunto']==split)&(metricas['Metrica_evaluada']=='Delay reconstruido')]
                    if len(q)!=1:
                        pendientes.append(f'Ganador {corner}/{target}/{split}: se esperaban 1 métrica, hay {len(q)}.')
                        continue
                    r=q.iloc[0]
                    tabla.append(_registro('modelo seleccionado por Validacion',model,params,
                       corner,target,split,r,n=int(r['N_evaluacion']),clasificacion='Ganador barrido general'))
        # Para RF selección concreta incluso si hay otro ganador en el barrido global.
        for corner in CORNERS:
            for target in ('Delay','Delta'):
                rank=_leer(path_general/f'ranking_{corner.lower()}_{target.lower()}.csv')
                if rank.empty:
                    pendientes.append(f'No está ranking_{corner.lower()}_{target.lower()}.csv')
                    continue
                if len(rank)!=49 or rank.duplicated(['Modelo','Experimento']).any():
                    raise RuntimeError(f'Ranking incompleto/duplicado: {corner}/{target}')
                # Auditar de forma explícita la coincidencia entre ranking y métricas.
                win_rf=rank.loc[rank.Modelo=='Random Forest'].sort_values('RMSE_Delay_Validacion').iloc[0]
                filtro=((metricas.Modelo=='Random Forest') & (metricas.Esquina==corner) &
                   (metricas.Objetivo==target) & (metricas.Experimento==win_rf['Experimento']) &
                   (metricas.Conjunto=='Validacion') & (metricas.Metrica_evaluada=='Delay reconstruido'))
                q=metricas.loc[filtro]
                if len(q)!=1:
                    raise RuntimeError(f'RF seleccionado {corner}/{target}: falta Validación exacta.')
                r=q.iloc[0]
                if not np.isclose(r['RMSE'],win_rf['RMSE_Delay_Validacion'],atol=1e-10):
                    raise RuntimeError(f'Ranking y métricas inconsistentes: {corner}/{target}')
                tabla.append(_registro('mejor RF entre 49 candidatos', 'Random Forest',win_rf['Experimento'],
                      corner,target,'Validacion',r,n=int(r['N_evaluacion']),clasificacion='Mejor RF Validación'))
        # Registrar blends de la ejecución: no cambiar sus parámetros ni mezclarlos con Delta.
        for _,r in metricas.loc[(metricas.Modelo=='Baseline-aware blend') &
              (metricas.Metrica_evaluada=='Delay reconstruido')].iterrows():
            tabla.append(_registro('baseline-aware',r['Modelo'],r['Experimento'],r['Esquina'],
                   r['Objetivo'],r['Conjunto'],r,n=int(r['N_evaluacion']),clasificacion='Blend validación'))
    for _,r in original_clf.iterrows():
        tabla.append(_registro('clasificador+regresor original', 'Clasificador + Regresor original',
              json.dumps({'tolerancia':float(r['Tolerancia']),
                          'classifier':'RandomForestClassifier','n_estimators_classifier':200,
                          'min_samples_leaf_classifier':2,
                          'regressor':'RandomForestRegressor','n_estimators_regressor':200,
                          'min_samples_leaf_regressor':2},sort_keys=True),
              r['Esquina'],'Delta',r['Conjunto'],{k:r[f'{k}_Delay'] for k in METRICAS},
              n=int(r['N']),clasificacion='Referencia 80% Train'))
    for _,r in hybrid_ref.iterrows():
        if r['Modelo'] not in ('Hybrid Model','Random Forest'): continue
        modelo=('Hybrid Model (referencia hoja 4)' if r['Modelo']=='Hybrid Model'
                else 'RF (referencia 100 arboles)')
        params=({'min_samples_leaf':4,'max_depth':9,'max_features':15,'LR_type':0}
                if r['Modelo']=='Hybrid Model' else
                {'n_estimators':100,'max_depth':20,'min_samples_leaf':4})
        tabla.append(_registro('comparacion híbrida de referencia',modelo,json.dumps(params,sort_keys=True),
           'Fast','Delta',r['Conjunto'],{k:r[f'{k}_Delay'] for k in METRICAS},
           clasificacion='Referencia histórica dentro de esta corrida'))
    for origen,hoja in [('barrido híbrido modo completo',hoja_original),('barrido híbrido nuevo',hoja_nueva)]:
        if hoja.empty:continue
        esperado={(i,j) for i in (1,2,4,8,16) for j in ('Validacion','Test Labels','Test Designs')}
        # La tabla nueva y la original poseen esquemas distintos. Se registra solo si hay métricas completas.
        for _,r in hoja.iterrows():
            params=(r['Parametros'] if 'Parametros' in r.index else json.dumps({'min_samples_leaf':int(r['min_samples_leaf'])}))
            vals={k:r.get(f'{k}_Delay',np.nan) for k in METRICAS}
            tabla.append(_registro(origen,'Hybrid Model (hojas barrido)',params,'Fast','Delta',r['Conjunto'],vals,
                                 n=r.get('N_evaluacion',np.nan),
                                 clasificacion='Seleccion por Validacion' if r.get('Criterio') else 'Candidato'))
    for _,r in cls_sel.iterrows():
        tabla.append(_registro('clasificador nuevo Validacion',r['Modelo'],r['Parametros'],
                  r['Esquina'],'Delta','Validacion', {k:r[f'{k}_Delay'] for k in METRICAS},
                  n=int(r['N_evaluacion']),clasificacion='Clasificador seleccionado'))
    for _,r in cls_test.iterrows():
        tabla.append(_registro('clasificador nuevo Test',r['Modelo'],r['Parametros'],
                 r['Esquina'],'Delta',r['Conjunto'],{k:r[f'{k}_Delay'] for k in METRICAS},
                 n=int(r['N_evaluacion']),clasificacion='Clasificador seleccionado'))
    tabla=pd.DataFrame(tabla)
    clave=['Origen','Esquina','Objetivo_entrenamiento','Modelo','Parametros','Conjunto','Metrica_evaluada','Seleccion']
    if tabla.duplicated(clave).any():
        bad=tabla.loc[tabla.duplicated(clave,keep=False),clave]
        raise RuntimeError('La tabla nueva contiene identidades duplicadas:\n'+bad.to_string(index=False))
    base=tabla.loc[(tabla.Modelo=='Baseline Typical') & (tabla.Conjunto.isin(SPLITS)),
         ['Esquina','Conjunto','MAE','RMSE']].rename(columns={'MAE':'MAE_Baseline','RMSE':'RMSE_Baseline'})
    tabla=tabla.merge(base,on=['Esquina','Conjunto'],how='left',validate='many_to_one')
    tabla['Mejora_RMSE_pct']=np.where(tabla.RMSE_Baseline>0,
           100*(tabla.RMSE_Baseline-tabla.RMSE)/tabla.RMSE_Baseline,np.nan)
    tabla['Mejora_MAE_pct']=np.where(tabla.MAE_Baseline>0,
           100*(tabla.MAE_Baseline-tabla.MAE)/tabla.MAE_Baseline,np.nan)
    tabla['Datos_verificados']='Mismos nueve CSV y mismo split de esta ejecución'
    tabla.to_csv(salida/'tabla_4_conjuntos.csv',index=False,float_format='%.16g')
    tabla.to_csv(salida/'resumen_para_reunion.csv',index=False,float_format='%.16g')
    comparacion=tabla[tabla.Conjunto.isin(['Validacion','Test Labels','Test Designs'])].copy()
    comparacion.to_csv(salida/'comparacion_baseline_RF_hibrido_clasificador.csv',
                      index=False,float_format='%.16g')
    # Alias solicitado por el equipo: misma corrida y métricas, sin datos históricos.
    comparacion.to_csv(salida/'comparacion_baseline_RF_hibrido.csv',
                      index=False,float_format='%.16g')
    # Resolver el filtro antiguo Fast Delta: los test usan 'Fast Delta (barrido)'.
    if not metricas.empty:
        q=metricas[(metricas.Esquina=='Fast') &
            (metricas.Modelo.isin(['Gradient Boosting','Hist Gradient Boosting'])) &
            (metricas.Objetivo.isin(['Delta','Fast Delta (barrido)'])) &
            (metricas.Conjunto.isin(['Validacion','Test Labels','Test Designs'])) &
            (metricas.Experimento.astype(str).str.contains('min_samples_leaf',regex=False))]
        q=q[((q.Conjunto=='Validacion')&(q.Objetivo=='Delta'))|
              ((q.Conjunto!='Validacion')&(q.Objetivo=='Fast Delta (barrido)'))]
        q=q.drop_duplicates(['Esquina','Modelo','Experimento','Conjunto','Metrica_evaluada'])
        q.to_csv(salida/'comparacion_min_samples_leaf_fast_delta.csv',index=False,float_format='%.16g')
        expected=5*2*3*2  # 5 hojas x 2 modelos x 3 conjuntos x Delta/Delay
        if len(q)!=expected:
            pendientes.append(f'Gradient Boosting barrido: esperadas {expected} filas, encontradas {len(q)}.')
    salida_pend=salida/'metricas_pendientes.csv'
    pd.DataFrame({'Pendiente':pendientes}).to_csv(salida_pend,index=False)
    resumen={'filas_tabla':len(tabla),'pendientes':pendientes,'completo':not pendientes}
    (salida/'resumen_verificacion.json').write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding='utf8')
    if pendientes and not permitir_parcial:
        raise RuntimeError('La tabla no está completa: '+'; '.join(pendientes))
    return resumen
