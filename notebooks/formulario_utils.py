"""
Utilidades compartidas para parsear el formulario de registro de tomas
de hidrocortisona ("Tomas hidrocortisona.xlsx"). Usado por
Coverage(t).ipynb y n_of_1_preprocesamiento.ipynb
(ventanas de exclusión de dosis para sigma_global).
"""

from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import numpy as np
import json

FORM_SHEET = "Respuestas de formulario 2"

COL_TIMESTAMP = "Marca temporal"
COL_DOSE = "De cuanto es la dosis?"
COL_TIME_OVERRIDE = "Actualizar la hora?"
COL_DATE_OVERRIDE = "Actualizar la fecha?"
COL_DOSE_TYPE = "Es dosis doble o normal?"

SLOT_LABELS = ["mañana", "mediodía", "tarde"]


def _resolve_datetime(row, base_ts_col, time_override_col, date_override_col):
    base_ts = row[base_ts_col]
    date_part = row[date_override_col].date() if pd.notna(row.get(date_override_col)) else base_ts.date()
    time_part = row[time_override_col] if pd.notna(row.get(time_override_col)) else base_ts.time()
    return pd.Timestamp.combine(date_part, time_part)


def _kmeans_1d(values: np.ndarray, k: int = 3, n_iter: int = 50) -> np.ndarray:
    """K-means en 1D, sin dependencias externas (evita requerir sklearn).
    Init por cuantiles equiespaciados; converge en pocas iteraciones dado
    que los datos son trimodales por diseño clínico (10-5-5)."""
    values = np.sort(values)
    centroids = np.quantile(values, np.linspace(0, 1, k, endpoint=False) + 1 / (2 * k))
    for _ in range(n_iter):
        assignments = np.argmin(np.abs(values[:, None] - centroids[None, :]), axis=1)
        new_centroids = np.array([
            values[assignments == j].mean() if (assignments == j).any() else centroids[j]
            for j in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return np.sort(centroids)


def fit_dose_slots(doses: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Deriva las k franjas horarias esperadas y su mg típico (mediana) a
    partir de las tomas 'Normal' históricas. Excluye deliberadamente los
    eventos is_extra=True (Doble) para no distorsionar los centros."""
    normal = doses[~doses["is_extra"]].copy()
    minutes = normal["datetime"].dt.hour * 60 + normal["datetime"].dt.minute + normal["datetime"].dt.second / 60
    centroids = _kmeans_1d(minutes.to_numpy(), k=k)
    assignments = np.argmin(np.abs(minutes.to_numpy()[:, None] - centroids[None, :]), axis=1)
    normal = normal.assign(_slot_idx=assignments)

    slots = []
    for j in range(k):
        subset = normal[normal["_slot_idx"] == j]
        slots.append({
            "slot_idx": j,
            "label": SLOT_LABELS[j] if k == len(SLOT_LABELS) else f"franja_{j}",
            "center_minutes": centroids[j],
            "typical_mg": subset["dose_mg"].median(),
            "n_obs": len(subset),
        })
    return pd.DataFrame(slots)


def _assign_slot(minute: float, centroids: np.ndarray) -> int:
    return int(np.argmin(np.abs(centroids - minute)))

CONFIRMED_OMISSIONS_PATH = Path(__file__).parent / "confirmed_omissions.json"


def _load_confirmed_omissions(path: Path = CONFIRMED_OMISSIONS_PATH) -> set:
    """Carga las fechas de omisión/reordenamiento confirmadas desde el JSON
    sidecar, para que todos los notebooks compartan la misma lista sin
    tener que repetirla. Si el archivo no existe, se asume que no hay
    ninguna confirmada todavía (no es un error)."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {pd.Timestamp(k).date() for k in data}


def detect_dose_gaps(doses: pd.DataFrame, slots: pd.DataFrame, mg_tolerance: float = 2.0,
                      confirmed_dates: set = frozenset()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Día a día, detecta franjas sin ninguna toma y tomas cuyo mg se desvía
    del típico de su franja (posible reordenamiento de pauta, p. ej. 07/18).
    No decide nada por sí mismo: separa el reporte en dos, según si la fecha
    del día está en `confirmed_dates` (declarada explícitamente por la autora
    como omisión real u reordenamiento confirmado, sin ninguna toma que añadir)
    o no (pendiente de revisión / posible fallo de registro sin resolver).

    Devuelve (pending_gaps, confirmed_gaps) -- misma estructura de columnas
    en ambos, separados solo por estado de confirmación."""
    centroids = slots["center_minutes"].to_numpy()
    normal = doses[~doses["is_extra"]].copy()
    normal["date"] = normal["datetime"].dt.date
    normal["minute"] = normal["datetime"].dt.hour * 60 + normal["datetime"].dt.minute + normal["datetime"].dt.second / 60
    normal["slot_idx"] = normal["minute"].apply(lambda m: _assign_slot(m, centroids))

    all_days = pd.date_range(doses["datetime"].min().date(), doses["datetime"].max().date(), freq="D").date
    gaps = []
    for day in all_days:
        day_rows = normal[normal["date"] == day]
        present_slots = set(day_rows["slot_idx"])
        for _, slot in slots.iterrows():
            j = int(slot["slot_idx"])
            if j not in present_slots:
                gaps.append({"date": day, "type": "missing_slot", "slot": slot["label"],
                             "expected_mg": slot["typical_mg"], "actual_mg": None, "actual_datetime": None,
                             "confirmed": day in confirmed_dates})
            else:
                match = day_rows[day_rows["slot_idx"] == j].iloc[0]
                if abs(match["dose_mg"] - slot["typical_mg"]) > mg_tolerance:
                    gaps.append({"date": day, "type": "magnitude_anomaly", "slot": slot["label"],
                                 "expected_mg": slot["typical_mg"], "actual_mg": match["dose_mg"],
                                 "actual_datetime": match["datetime"], "confirmed": day in confirmed_dates})
    gaps_df = pd.DataFrame(gaps)
    if gaps_df.empty:
        empty = pd.DataFrame(columns=["date", "type", "slot", "expected_mg", "actual_mg", "actual_datetime"])
        return empty, empty.copy()
    pending = gaps_df[~gaps_df["confirmed"]].drop(columns="confirmed").reset_index(drop=True)
    confirmed = gaps_df[gaps_df["confirmed"]].drop(columns="confirmed").reset_index(drop=True)
    return pending, confirmed


def load_form_doses(xlsx_path, sheet_name: str = FORM_SHEET, k_slots: int = 3, mg_tolerance: float = 2.0,
                     confirmed_omission_dates: set = None) -> pd.DataFrame:
    """[...]. Si se omite (None), se cargan automáticamente desde
    confirmed_omissions.json (mismo directorio que este módulo) -- fuente
    única compartida por todos los notebooks. Pasa un set explícito solo
    para pruebas puntuales o para sobreescribir el archivo."""
    if confirmed_omission_dates is None:
        confirmed_omission_dates = _load_confirmed_omissions()
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    col_map = {c.strip(): c for c in df.columns}
    def col(name): return col_map.get(name.strip(), name)

    scheduled = df[df[col(COL_DOSE)].notna() & (df[col(COL_DOSE)] > 0)].copy()
    events = []
    for _, row in scheduled.iterrows():
        dt = _resolve_datetime(row, col(COL_TIMESTAMP), col(COL_TIME_OVERRIDE), col(COL_DATE_OVERRIDE))
        is_extra = str(row.get(col(COL_DOSE_TYPE), "")).strip().lower() == "doble"
        events.append({"datetime": dt, "dose_mg": float(row[col(COL_DOSE)]), "is_extra": is_extra, "source": "scheduled_actual"})
    out = pd.DataFrame(events).sort_values("datetime").reset_index(drop=True)

    slots = fit_dose_slots(out, k=k_slots)
    pending, confirmed = detect_dose_gaps(out, slots, mg_tolerance=mg_tolerance,
                                           confirmed_dates=confirmed_omission_dates)
    out.attrs["dose_slots"] = slots
    out.attrs["pending_gaps"] = pending
    out.attrs["confirmed_gaps"] = confirmed

    if len(pending) > 0:
        print(f"AVISO: {pending['date'].nunique()} día(s) con inconsistencias pendientes de reconciliación manual.")
        print("Ver doses.attrs['pending_gaps']. Coverage(t) se calculará solo con los eventos ya presentes (Opción B).")
    if len(confirmed) > 0:
        print(f"INFO: {confirmed['date'].nunique()} día(s) con omisión/reordenamiento confirmado por la autora (doses.attrs['confirmed_gaps']), sin cobertura añadida.")

    return out
# ======================================================================================================
# EXtracción del "groundtruth" del formulario de dosis
# Usado por MVA_preprocesamiento.ipynb
# ======================================================================================================

import pandas as pd
import numpy as np



COL_STRESS_A = "Estrés percibido desde la última toma"
COL_SYMPTOMS_A = "¿Has notado alguno de estos síntomas desde la última toma?"
COL_CONTEXT_A = "¿Hubo algo relevante en el contexto desde la última toma?"
COL_SLEEP_A = "Calidad de sueño de anoche (solo si es la entrada de la mañana)"

COL_MOTIVO_B = "Motivo de registro"
COL_EVENT_END_B = "Hora aproximada de fin del evento"
COL_DURATION_B = "Duracion del mismo en minutos"
COL_STRESS_B = "Estrés percibido en ese momento"
COL_SYMPTOMS_B = "Síntomas presentes"
COL_SYMPTOM_CAUSE_B = "Qué síntomas provocaron"

MIDNIGHT_ROLLOVER_HOUR = 6  # umbral confirmado




def _resolve_event_end_date(row, ts_col, event_time_col, date_override_col, rollover_hour=MIDNIGHT_ROLLOVER_HOUR):
    """Resuelve la fecha del fin de evento de Bloque B. Prioridad: 1) 'Actualizar
    la fecha?' si está declarada, 2) heurística de cruce de medianoche (red de
    seguridad). Devuelve (datetime, era_inferido)."""
    ts = row[ts_col]
    event_time = row[event_time_col]
    if pd.isna(event_time):
        return None, None

    if pd.notna(row.get(date_override_col)):
        date_part = row[date_override_col].date()
        return pd.Timestamp.combine(date_part, event_time), False

    submission_minute = ts.hour * 60 + ts.minute
    event_minute = event_time.hour * 60 + event_time.minute
    if event_minute > submission_minute and ts.hour < rollover_hour:
        date_part = (ts - pd.Timedelta(days=1)).date()
        inferred = True
    else:
        date_part = ts.date()
        inferred = False
    return pd.Timestamp.combine(date_part, event_time), inferred


def load_form_ema(xlsx_path, doses: pd.DataFrame, sheet_name: str = FORM_SHEET) -> pd.DataFrame:
    """Extrae el ground truth EMA combinando Bloque A (reporte de intervalo,
    ligado a cada toma) y Bloque B (evento puntual con duración declarada,
    tratado ahora con la misma estructura de ventana que Bloque A) en una
    única tabla larga con columna `block`.
    Requiere el DataFrame de dosis ya cargado para calcular `window_start`
    de Bloque A y `window_reliable` de ambos bloques."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    col_map = {c.strip(): c for c in df.columns}
    def col(name): return col_map.get(name.strip(), name)

    dose_times = doses["datetime"].sort_values().reset_index(drop=True)
    pending_dates = set(doses.attrs.get("pending_gaps", pd.DataFrame(columns=["date"]))["date"])

    def _window_reliable(window_start, window_end):
        if pd.isna(window_start) or pd.isna(window_end):
            return False
        window_dates = set(pd.date_range(window_start.date(), window_end.date(), freq="D").date)
        return not (window_dates & pending_dates)

    records = []

    # --- Bloque A: reporte retrospectivo de intervalo, ligado a cada toma ---
    block_a = df[df[col(COL_STRESS_A)].notna()].copy()
    for _, row in block_a.iterrows():
        window_end = _resolve_datetime(row, col(COL_TIMESTAMP), col(COL_TIME_OVERRIDE), col(COL_DATE_OVERRIDE))
        prior = dose_times[dose_times < window_end]
        window_start = prior.iloc[-1] if len(prior) > 0 else pd.NaT
        records.append({
            "block": "A", "window_start": window_start, "window_end": window_end,
            "stress_score": row[col(COL_STRESS_A)], "symptoms": row.get(col(COL_SYMPTOMS_A)),
            "context_flag": row.get(col(COL_CONTEXT_A)), "sleep_quality": row.get(col(COL_SLEEP_A)),
            "window_reliable": _window_reliable(window_start, window_end),
        })

    # --- Bloque B: evento puntual con duración declarada -> ventana propia ---
    block_b = df[df[col(COL_MOTIVO_B)].notna()].copy()
    n_inferred = 0
    for _, row in block_b.iterrows():
        window_end, inferred = _resolve_event_end_date(
            row, col(COL_TIMESTAMP), col(COL_EVENT_END_B), col(COL_DATE_OVERRIDE)
        )
        if inferred:
            n_inferred += 1
        duration = row.get(col(COL_DURATION_B))
        window_start = window_end - pd.Timedelta(minutes=duration) if (window_end is not None and pd.notna(duration)) else pd.NaT

        context_parts = [str(row[col(COL_MOTIVO_B)])]
        cause = row.get(col(COL_SYMPTOM_CAUSE_B))
        if pd.notna(cause):
            context_parts.append(str(cause))
        context_flag = "; ".join(context_parts)

        records.append({
            "block": "B", "window_start": window_start, "window_end": window_end,
            "stress_score": row.get(col(COL_STRESS_B)), "symptoms": row.get(col(COL_SYMPTOMS_B)),
            "context_flag": context_flag, "sleep_quality": None,
            "window_reliable": _window_reliable(window_start, window_end),
        })

    out = pd.DataFrame(records)
    out = out.sort_values("window_end", na_position="last").reset_index(drop=True)

    n_unreliable = int((out["window_reliable"] == False).sum())
    if n_unreliable:
        print(f"AVISO: {n_unreliable} registro(s) (A+B) con ventana solapando huecos de dosis pendientes o sin resolver.")
    if n_inferred:
        print(f"AVISO: {n_inferred} evento(s) de Bloque B con fecha de fin inferida (no declarada) por heurística de medianoche.")

    return out

#Ejemplo de uso:
#doses = load_form_doses(ruta_excel)
#ema = load_form_ema(ruta_excel, doses)

# Para validación ecológica, filtrar explícitamente lo fiable:
#ema_valido = ema[(ema["block"] == "B") | (ema["window_reliable"] == True)]

# ======================================================================================================
# NÚCLEO PK COMPARTIDO — Coverage(t), modelo de Bateman
# Movido desde coverage_model_v1_3.ipynb para reutilizarlo en otros notebooks (p. ej. escenarios sintéticos)
# ======================================================================================================

def _tmax_of_ka(ka: float, ke: float) -> float:
    """tmax = ln(ka/ke)/(ka-ke), con el caso límite ka=ke -> tmax=1/ke."""
    if abs(ka - ke) < 1e-9:
        return 1.0 / ke
    return np.log(ka / ke) / (ka - ke)


def _solve_ka_from_tmax(ke: float, tmax_target: float, hi: float = 20.0, tol: float = 1e-10) -> float:
    """Resuelve por bisección el ka que, para un ke fijo, produce el tmax objetivo."""
    lo = ke * 1.0001
    f_lo = _tmax_of_ka(lo, ke) - tmax_target
    mid = lo
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = _tmax_of_ka(mid, ke) - tmax_target
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
        if abs(f_mid) < tol:
            break
    return mid


# Derendorf et al. (1991): t1/2 = 1.7 h; tmax observado 1-1.5 h (punto medio 1.25 h)
_KE_DEFAULT = np.log(2) / 1.7
_KA_DEFAULT = _solve_ka_from_tmax(_KE_DEFAULT, tmax_target=1.25)


@dataclass
class PKParams:
    ka: float = _KA_DEFAULT   # h^-1, constante de absorción — derivada, ver Coverage_t_.ipynb
    ke: float = _KE_DEFAULT   # h^-1, constante de eliminación — derivada de t1/2=1.7h

    @property
    def half_life_h(self) -> float:
        return np.log(2) / self.ke


def bateman_single_dose(t_rel: np.ndarray, dose_mg: float, pk: PKParams) -> np.ndarray:
    """Contribución de una única toma a Coverage(t), evaluada en tiempos
    relativos a la hora de la toma (t_rel en horas, puede incluir negativos:
    se fuerza a 0 antes de t=0)."""
    t_rel = np.asarray(t_rel, dtype=float)
    out = np.zeros_like(t_rel)
    mask = t_rel >= 0
    ka, ke = pk.ka, pk.ke
    if abs(ka - ke) < 1e-6:
        out[mask] = dose_mg * ka * t_rel[mask] * np.exp(-ke * t_rel[mask])
    else:
        out[mask] = dose_mg * (ka / (ka - ke)) * (
            np.exp(-ke * t_rel[mask]) - np.exp(-ka * t_rel[mask])
        )
    return out


def compute_coverage(
    dose_events: pd.DataFrame,
    time_grid: pd.DatetimeIndex,
    pk: PKParams = PKParams(),
    normalize: bool = False,
) -> pd.Series:
    """Superposición lineal de todas las tomas sobre una rejilla temporal.
    Ver docstring original en Coverage_t_.ipynb para el detalle completo
    de unidades y supuestos (mg-equivalente/h, no concentración; linealidad sin
    saturación de CBG)."""
    t_hours = (time_grid.values - time_grid.values[0]) / np.timedelta64(1, "h")
    coverage = np.zeros_like(t_hours, dtype=float)

    for _, ev in dose_events.iterrows():
        t_dose_h = (ev["datetime"] - time_grid[0]) / pd.Timedelta(hours=1)
        coverage += bateman_single_dose(t_hours - t_dose_h, ev["dose_mg"], pk)

    series = pd.Series(coverage, index=time_grid, name="coverage_mg_per_h")

    if normalize:
        ref = coverage.max()
        if ref > 0:
            series = series / ref  # SOLO visualización — nunca para Risk(t)

    return series


def build_standard_schedule(
    start_date: pd.Timestamp,
    n_days: int,
    times=("07:30", "13:00", "17:00"),
    doses=(10, 5, 5),
) -> pd.DataFrame:
    """Genera el calendario teórico de tomas 10-5-5 para n_days días."""
    assert len(times) == len(doses), "times y doses deben tener la misma longitud"
    rows = []
    for d in range(n_days):
        day = start_date + pd.Timedelta(days=d)
        for t_str, dose in zip(times, doses):
            hh, mm = map(int, t_str.split(":"))
            rows.append({
                "datetime": day + pd.Timedelta(hours=hh, minutes=mm),
                "dose_mg": float(dose),
                "source": "scheduled_standard",
            })
    return pd.DataFrame(rows)

# ======================================================================================================
# NÚCLEO DEMAND_CIRCADIAN(T) — Dual Cosines (Chakraborty, Krzyzanski & Jusko, 1999)
# Movido desde Demanda_v2.ipynb para reutilizarlo en otros notebooks (p. ej. escenarios sintéticos)
# ======================================================================================================

# Pico y nadir reales reportados por Debono et al. (2009) — fijan Rm y Ramp
_PICO_DEBONO = 15.5
_T_MAX_DEBONO = 8 + 32/60        # 08:32h — hora del pico (acrofase)
_NADIR_DEBONO = 2.0
_T_MIN_DEBONO = 0 + 18/60        # 00:18h — hora del nadir

RM_DEMAND = (_PICO_DEBONO + _NADIR_DEBONO) / 2    # nivel medio de la función Dual Cosines
RAMP_DEMAND = (_PICO_DEBONO - _NADIR_DEBONO) / 2  # amplitud de la función Dual Cosines
T_MIN_DEMAND = _T_MIN_DEBONO
T_MAX_DEMAND = _T_MAX_DEBONO

# Producción endógena fisiológica diaria (Chan & Debono, 2010) — magnitud de referencia
DAILY_TOTAL_MG_MID = np.mean([9.5, 9.9])  # 9.7 mg/día


def dual_cosines(t_horas, Rm=RM_DEMAND, Ramp=RAMP_DEMAND, T_min=T_MIN_DEMAND, T_max=T_MAX_DEMAND):
    """Método Dual Cosines (Rohatagi et al. 1996; Ecs. 12-14, Chakraborty, Krzyzanski & Jusko, 1999).
    Dos cosenos de periodo distinto (subida y bajada) empalmados en T_min y T_max.
    Acotado por construcción en [Rm-Ramp, Rm+Ramp] — nunca negativo si Rm >= Ramp."""
    t = np.asarray(t_horas, dtype=float) % 24
    out = np.empty_like(t)

    subida = (t >= T_min) & (t <= T_max)
    out[subida] = Rm + Ramp * np.cos(2*np.pi*(t[subida] - 2*T_min + T_max) / (2*(T_max - T_min)))

    bajada_tarde = t > T_max
    out[bajada_tarde] = Rm + Ramp * np.cos(2*np.pi*(t[bajada_tarde] - T_max) / (2*(T_min - T_max + 24)))

    bajada_temprana = t < T_min
    out[bajada_temprana] = Rm + Ramp * np.cos(2*np.pi*(t[bajada_temprana] + 24 - T_max) / (2*(T_min - T_max + 24)))

    return out


def demand_circadian(t_horas, daily_total_mg=DAILY_TOTAL_MG_MID,
                      Rm=RM_DEMAND, Ramp=RAMP_DEMAND, T_min=T_MIN_DEMAND, T_max=T_MAX_DEMAND):
    """Demand_circadian(t) en mg-equivalentes/hora — fase poblacional (Debono et al. 2009),
    usada en escenarios sintéticos. Para el caso n-of-1, ver demand_circadian_personalizado()
    en Demanda_v2.ipynb (desplaza T_min/T_max a la hora de despertar real de la autora)."""
    forma = dual_cosines(t_horas, Rm, Ramp, T_min, T_max)
    tref = np.linspace(0, 24, 24*60, endpoint=False)
    forma_ref = dual_cosines(tref, Rm, Ramp, T_min, T_max)
    integral_forma = np.trapezoid(forma_ref, tref)
    return forma * (daily_total_mg / integral_forma)


D_REF_MID = 0.57  # mg/h — Kirschbaum et al. (1993), ya cerrado (Demanda_v2.ipynb, Sección 2)

def demand_circadian_personalizado(t_horas, daily_total_mg, hora_despertar_media,
                                    Rm=None, Ramp=None, T_min=None, T_max=None,
                                    desfase_pico_tras_despertar_h=0.0):
    """Igual que demand_circadian(), pero T_min/T_max se desplazan en bloque para que el pico
    (T_max) caiga en la hora de despertar real (+ desfase opcional, no investigado, 0.0 por defecto).
    Usado en la validación n-of-1 (ver n_of_1_calculo_validacion.ipynb)."""
    Rm = RM_DEMAND if Rm is None else Rm
    Ramp = RAMP_DEMAND if Ramp is None else Ramp
    T_min = T_MIN_DEMAND if T_min is None else T_min
    T_max = T_MAX_DEMAND if T_max is None else T_max
    offset = (hora_despertar_media + desfase_pico_tras_despertar_h) - T_max
    return demand_circadian(t_horas, daily_total_mg, Rm=Rm, Ramp=Ramp,
                             T_min=T_min + offset, T_max=T_max + offset)


# ======================================================================================================
# VARIANTES DE PATRÓN DE DOSIFICACIÓN — retraso u omisión de una toma concreta del schedule estándar
# ======================================================================================================
def aplicar_variante_dosis(schedule, fecha, hora_str, tipo, retraso_horas=2.0):
    """
    Modifica UNA toma concreta de un schedule ya generado por build_standard_schedule(),
    identificada por fecha exacta + hora ('07:30'), sin tocar el resto de tomas.

    tipo: 'a_tiempo' (sin cambios, devuelve el schedule tal cual) | 'retrasada' (desplaza
    esa toma retraso_horas más tarde) | 'omitida' (elimina esa toma del schedule).
    """
    schedule = schedule.copy()
    hh, mm = map(int, hora_str.split(":"))
    momento_dosis = fecha + pd.Timedelta(hours=hh, minutes=mm)
    mask = schedule['datetime'] == momento_dosis

    if mask.sum() != 1:
        raise ValueError(f"Se esperaba encontrar exactamente 1 toma en {momento_dosis}, "
                          f"se encontraron {mask.sum()} — revisar fecha/hora.")

    if tipo == 'a_tiempo':
        return schedule
    elif tipo == 'omitida':
        return schedule[~mask].reset_index(drop=True)
    elif tipo == 'retrasada':
        schedule.loc[mask, 'datetime'] = schedule.loc[mask, 'datetime'] + pd.Timedelta(hours=retraso_horas)
        return schedule
    else:
        raise ValueError(f"tipo debe ser 'a_tiempo', 'retrasada' u 'omitida' — recibido: {tipo}")
