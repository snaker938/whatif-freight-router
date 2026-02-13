export type Locale = 'en' | 'es';

export const LOCALE_OPTIONS: Array<{ value: Locale; label: string }> = [
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Español' },
];

const DICTIONARY = {
  en: {
    skip_to_controls: 'Skip to controls panel',
    panel_title: 'Carbon-Aware Freight Router',
    panel_subtitle:
      'Click the map to set Start, then End. Compute Pareto to generate candidate routes, then use the sliders to choose the best trade-off (time vs cost vs CO2).',
    setup: 'Setup',
    language: 'Language',
    start_tutorial: 'Start tutorial',
    preferences: 'Preferences',
    compute_pareto: 'Compute Pareto',
    computing: 'Computing...',
    clear_results: 'Clear results',
    compare_scenarios: 'Compare Scenarios',
    comparing_scenarios: 'Comparing Scenarios...',
    route_hint_start: 'Click the map to set Start.',
    route_hint_destination: 'Now click the map to set End.',
    route_hint_default:
      'Compute Pareto to compare candidate routes. Use sliders to pick the best trade-off.',
    live_computing_progress: 'Computing routes {done} of {total}',
    live_computing: 'Computing routes',
    live_compare_complete: 'Scenario comparison complete',
    live_compare_failed: 'Scenario comparison failed',
    live_departure_complete: 'Departure optimization complete',
    live_departure_failed: 'Departure optimization failed',
    live_duty_complete: 'Duty chain complete',
    live_duty_failed: 'Duty chain failed',
    map_overlays: 'Map overlays',
    overlay_stops: 'Stops',
    overlay_incidents: 'Incidents',
    overlay_segments: 'Segments',
    incident_dwell: 'Dwell',
    incident_accident: 'Accident',
    incident_closure: 'Closure',
    segment_label: 'Segment',
    stop_label: 'Stop',
  },
  es: {
    skip_to_controls: 'Saltar al panel de controles',
    panel_title: 'Router de Carga con Conciencia de Carbono',
    panel_subtitle:
      'Haz clic en el mapa para definir inicio y destino. Calcula Pareto para generar rutas candidatas y usa los controles para elegir el mejor equilibrio.',
    setup: 'Configuración',
    language: 'Idioma',
    start_tutorial: 'Iniciar tutorial',
    preferences: 'Preferencias',
    compute_pareto: 'Calcular Pareto',
    computing: 'Calculando...',
    clear_results: 'Limpiar resultados',
    compare_scenarios: 'Comparar Escenarios',
    comparing_scenarios: 'Comparando Escenarios...',
    route_hint_start: 'Haz clic en el mapa para definir el inicio.',
    route_hint_destination: 'Ahora haz clic para definir el fin.',
    route_hint_default:
      'Calcula Pareto para comparar rutas candidatas. Ajusta los controles para el mejor equilibrio.',
    live_computing_progress: 'Calculando rutas {done} de {total}',
    live_computing: 'Calculando rutas',
    live_compare_complete: 'Comparación de escenarios completada',
    live_compare_failed: 'La comparación de escenarios falló',
    live_departure_complete: 'Optimización de salida completada',
    live_departure_failed: 'La optimización de salida falló',
    live_duty_complete: 'Cadena de servicio completada',
    live_duty_failed: 'La cadena de servicio falló',
    map_overlays: 'Capas del mapa',
    overlay_stops: 'Paradas',
    overlay_incidents: 'Incidentes',
    overlay_segments: 'Segmentos',
    incident_dwell: 'Espera',
    incident_accident: 'Accidente',
    incident_closure: 'Cierre',
    segment_label: 'Segmento',
    stop_label: 'Parada',
  },
} as const;

type DictionaryKey = keyof (typeof DICTIONARY)['en'];

function replaceTokens(template: string, params?: Record<string, string | number>): string {
  if (!params) return template;
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, key) => {
    if (!(key in params)) return match;
    return String(params[key]);
  });
}

export function createTranslator(locale: Locale) {
  return (key: DictionaryKey, params?: Record<string, string | number>): string => {
    const dictionary = DICTIONARY[locale] ?? DICTIONARY.en;
    const template = dictionary[key] ?? DICTIONARY.en[key];
    return replaceTokens(template, params);
  };
}
