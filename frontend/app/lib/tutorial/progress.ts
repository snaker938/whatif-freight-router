import type { TutorialProgress } from './types';

export function loadTutorialProgress(progressKey: string): TutorialProgress | null {
  try {
    const raw = window.localStorage.getItem(progressKey);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<TutorialProgress>;
    if (!parsed || typeof parsed.stepIndex !== 'number' || !Array.isArray(parsed.actions)) {
      return null;
    }
    return {
      stepIndex: parsed.stepIndex,
      actions: parsed.actions.filter((item): item is string => typeof item === 'string'),
      optionalDecisions: Array.isArray(parsed.optionalDecisions)
        ? parsed.optionalDecisions.filter((item): item is string => typeof item === 'string')
        : [],
      updatedAt: typeof parsed.updatedAt === 'string' ? parsed.updatedAt : new Date().toISOString(),
    };
  } catch {
    return null;
  }
}

export function saveTutorialProgress(progressKey: string, progress: TutorialProgress): void {
  try {
    window.localStorage.setItem(progressKey, JSON.stringify(progress));
  } catch {
    // Ignore write errors.
  }
}

export function clearTutorialProgress(progressKey: string): void {
  try {
    window.localStorage.removeItem(progressKey);
  } catch {
    // Ignore write errors.
  }
}

export function loadTutorialCompleted(completedKey: string): boolean {
  try {
    return window.localStorage.getItem(completedKey) === '1';
  } catch {
    return false;
  }
}

export function saveTutorialCompleted(completedKey: string, completed: boolean): void {
  try {
    if (completed) {
      window.localStorage.setItem(completedKey, '1');
    } else {
      window.localStorage.removeItem(completedKey);
    }
  } catch {
    // Ignore write errors.
  }
}
