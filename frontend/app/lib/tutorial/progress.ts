import type { TutorialProgress } from './types';

export function loadTutorialProgress(progressKey: string): TutorialProgress | null {
  try {
    const raw = window.localStorage.getItem(progressKey);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<TutorialProgress>;
    if (!parsed || parsed.version !== 3 || typeof parsed.stepIndex !== 'number') {
      window.localStorage.removeItem(progressKey);
      return null;
    }
    const stepActionsById =
      parsed.stepActionsById && typeof parsed.stepActionsById === 'object'
        ? Object.entries(parsed.stepActionsById).reduce<Record<string, string[]>>((acc, [stepId, actions]) => {
            if (!Array.isArray(actions)) return acc;
            const filtered = actions.filter((item): item is string => typeof item === 'string');
            acc[stepId] = filtered;
            return acc;
          }, {})
        : {};
    const optionalDecisionsByStep =
      parsed.optionalDecisionsByStep && typeof parsed.optionalDecisionsByStep === 'object'
        ? Object.entries(parsed.optionalDecisionsByStep).reduce<Record<string, string[]>>(
            (acc, [stepId, decisions]) => {
              if (!Array.isArray(decisions)) return acc;
              const filtered = decisions.filter((item): item is string => typeof item === 'string');
              acc[stepId] = filtered;
              return acc;
            },
            {},
          )
        : {};
    return {
      version: 3,
      stepIndex: parsed.stepIndex,
      stepActionsById,
      optionalDecisionsByStep,
      updatedAt: typeof parsed.updatedAt === 'string' ? parsed.updatedAt : new Date().toISOString(),
    };
  } catch {
    try {
      window.localStorage.removeItem(progressKey);
    } catch {
      // Ignore write errors.
    }
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
