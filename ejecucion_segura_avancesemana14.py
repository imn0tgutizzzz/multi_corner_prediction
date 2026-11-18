"""Reanudación y control de integridad para el TFG multicorner.

Un checkpoint solo se reutiliza cuando el manifiesto de entradas, código,
parámetros, divisiones y versiones coincide exactamente con la corrida.
Nunca se mezclan archivos de corridas distintas.
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
import joblib


def sha256_archivo(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloque)
    return h.hexdigest()


def _json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str)


def escritura_json_atomica(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporal=path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporal.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        os.replace(temporal,path)
    finally:
        if temporal.exists(): temporal.unlink()


def crear_identidad(api, raiz_codigo, datos, *, n_jobs, backend, modo):
    import sklearn, numpy, pandas
    train, tests, features, X, Xt, i_train, i_val = datos
    raiz=Path(raiz_codigo)
    entrada_dir=Path(api.BASE_DIR)
    modulos=('modelo_objetivo3_baseline_aware_integrado.py',
             'experimentos_integrados_paso4.py','tablas_historicas.py',
             'ejecucion_segura.py','tablas_actuales.py','hybridmodel_min_samples_leaf.py')
    fuentes={p:sha256_archivo(raiz/p) for p in modulos}
    entradas={}
    for pre in ('treated_labels_train','treated_labels','treated_test_designs'):
        for corner in ('typical','slow','fast'):
            nombre=f'{pre}_{corner}.csv'
            fichero=entrada_dir/nombre
            if not fichero.is_file():
                raise FileNotFoundError(f'Falta uno de los nueve CSV: {fichero}')
            entradas[nombre]=sha256_archivo(fichero)
    import xgboost
    ident={
        'formato':3,'modo':modo, 'codigo_sha256':fuentes,'csv_sha256':entradas,
        'versiones':{'python':sys.version.split()[0], 'sklearn':sklearn.__version__,
          'numpy':numpy.__version__,'pandas':pandas.__version__,
          'joblib':joblib.__version__,'xgboost':xgboost.__version__},
        'parametros':{'random_state':api.RANDOM_STATE, 'svr_max_train':api.SVR_MAX_TRAIN,
          'n_jobs':n_jobs,'backend':backend,'tolerancia':.05,
          'hybrid_hojas':list(api.HYBRID_MIN_SAMPLES_LEAF_EXPERIMENTS),
          'configuraciones':[(m,p) for m,p in api.crear_experimentos()]},
        'indices_sha256':{'train':hashlib.sha256(i_train.astype('int64').tobytes()).hexdigest(),
                 'validacion':hashlib.sha256(i_val.astype('int64').tobytes()).hexdigest()},
        'features':features,'filas':{'Train':len(train), 'Test Labels':len(tests['Test Labels']),
                                      'Test Designs':len(tests['Test Designs'])},
    }
    ident['fingerprint']=hashlib.sha256(_json(ident).encode()).hexdigest()
    return ident


class GestorEjecucion:
    def __init__(self, ruta, identidad, *, reanudar=False):
        self.ruta=Path(ruta).resolve()
        self.control=self.ruta/'control_ejecucion'
        self.control.mkdir(parents=True,exist_ok=True)
        self.checkpoints=self.control/'checkpoints'
        self.checkpoints.mkdir(exist_ok=True)
        self.manifiesto=self.control/'manifiesto.json'
        self.estado=self.control/'estado.json'
        if reanudar:
            if not self.manifiesto.is_file():
                raise FileNotFoundError(f'No hay manifiesto para reanudar: {self.manifiesto}')
            anterior=json.loads(self.manifiesto.read_text(encoding='utf-8'))
            if anterior['fingerprint']!=identidad['fingerprint']:
                raise ValueError('REANUDACIÓN RECHAZADA: cambiaron CSV, código, versiones, '
                       'split, parámetros, modo o paralelismo. Inicie una ejecución nueva.')
        else:
            if self.manifiesto.exists():
                raise FileExistsError('La carpeta ya contiene una ejecución; use --reanudar.')
            escritura_json_atomica(self.manifiesto,identidad)
        self.identidad=identidad
        self.registrar('CORRIDA','REANUDADA' if reanudar else 'INICIADA',{'modo':identidad['modo']})

    def _paths(self, key):
        h=hashlib.sha256(str(key).encode('utf-8')).hexdigest()
        return self.checkpoints/(h+'.joblib'),self.checkpoints/(h+'.json')

    def cargar(self,key):
        p,meta=self._paths(key)
        if not p.exists() and not meta.exists(): return None
        if not p.exists() or not meta.exists():
            # Un corte de luz entre escribir el archivo y su meta es posible.
            # Ese experimento NO cuenta como terminado y se vuelve a entrenar.
            self.registrar(key,'CHECKPOINT_INCOMPLETO_REENTRENAR',{'archivo':str(p),'meta':str(meta)})
            print(f'ADVERTENCIA: checkpoint incompleto ({key}). Se repetirá SOLO ese experimento.')
            return None
        info=json.loads(meta.read_text(encoding='utf-8'))
        if info['key']!=str(key) or info['fingerprint']!=self.identidad['fingerprint']:
            raise RuntimeError(f'Checkpoint de otra ejecución ({key})')
        if sha256_archivo(p)!=info['sha256']:
            raise RuntimeError(f'Checkpoint corrupto ({key}); no se reutilizará.')
        return joblib.load(p)

    def guardar(self,key,payload,*,contexto=None):
        p,meta=self._paths(key)
        temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            joblib.dump(payload,temp,compress=2)
            digest=sha256_archivo(temp)
            os.replace(temp,p)
            escritura_json_atomica(meta,{'key':str(key),'sha256':digest,
                      'fingerprint':self.identidad['fingerprint'],
                      'contexto':contexto or {}, 'completado':datetime.now(timezone.utc).isoformat()})
            self.registrar(key,'COMPLETADO',contexto or {})
        finally:
            if temp.exists():temp.unlink()

    def registrar(self,key,estado,datos=None):
        if estado == 'INICIADO' and key != 'CORRIDA':
            self.experimento_activo=str(key)
        elif estado in ('COMPLETADO','VERIFICADO') and getattr(self,'experimento_activo',None)==str(key):
            self.experimento_activo=None
        entrada={'fecha_utc':datetime.now(timezone.utc).isoformat(),
                 'experimento':str(key),'estado':estado,'detalle':datos or {}}
        journal=self.control/'eventos.jsonl'
        with open(journal,'a',encoding='utf-8') as out:
            out.write(_json(entrada)+'\n')
            out.flush();os.fsync(out.fileno())
        escritura_json_atomica(self.estado,{'ultimo_evento':entrada,
             'estado_corrida':estado if key=='CORRIDA' else 'EN_PROGRESO'})

    def comprobar(self,key,checks):
        faltan=[str(x) for x in checks if not Path(x).is_file()]
        if faltan:
            self.registrar(key,'FALLIDO',{'faltan':faltan})
            raise RuntimeError(f'Faltan archivos obligatorios al terminar {key}: {faltan}')
        self.registrar(key,'VERIFICADO',{'archivos':len(checks)})

    def fallo(self,key,exc):
        activo=getattr(self,'experimento_activo',None)
        self.registrar(key,'FALLIDO',{'tipo':type(exc).__name__,
                   'mensaje':str(exc),'experimento_activo':activo})
        if activo:
            print(f'EXPERIMENTO QUE ESTABA EN CURSO: {activo}',file=sys.stderr)

    def terminar(self):
        self.registrar('CORRIDA','TERMINADA',{'fingerprint':self.identidad['fingerprint']})
        escritura_json_atomica(self.control/'ejecucion_completa.json',
              {'fingerprint':self.identidad['fingerprint'],'terminada':True,
               'fecha_utc':datetime.now(timezone.utc).isoformat()})
