import type { LatLng } from '../types';
import type { TutorialPrefillId } from './types';

export const TUTORIAL_PROGRESS_KEY = 'tutorial_v2_progress';
export const TUTORIAL_COMPLETED_KEY = 'tutorial_v2_completed';

export const TUTORIAL_CANONICAL_ORIGIN: LatLng = {
  lat: 52.4862,
  lon: -1.8904,
};

export const TUTORIAL_CANONICAL_DESTINATION: LatLng = {
  lat: 51.5072,
  lon: -0.1276,
};

export const TUTORIAL_CANONICAL_DUTY_STOPS = [
  '52.4862,-1.8904,Birmingham',
  '52.2053,0.1218,Cambridge',
  '51.5072,-0.1276,London',
].join('\n');

export function nextUtcHourLocalInput(now = new Date()): string {
  const dt = new Date(now);
  dt.setUTCMinutes(0, 0, 0);
  dt.setUTCHours(dt.getUTCHours() + 1);
  return dt.toISOString().slice(0, 16);
}

export function defaultDepartureWindow(now = new Date()): {
  start: string;
  end: string;
} {
  const start = new Date(now.getTime() + 30 * 60 * 1000);
  const end = new Date(start.getTime() + 6 * 60 * 60 * 1000);
  return {
    start: start.toISOString().slice(0, 16),
    end: end.toISOString().slice(0, 16),
  };
}

export function isTutorialPrefillId(value: string): value is TutorialPrefillId {
  return (
    value === 'canonical_map' ||
    value === 'canonical_setup' ||
    value === 'canonical_advanced' ||
    value === 'canonical_preferences' ||
    value === 'canonical_departure' ||
    value === 'canonical_duty' ||
    value === 'canonical_oracle' ||
    value === 'canonical_experiment'
  );
}
