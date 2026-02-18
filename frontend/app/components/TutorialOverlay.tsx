'use client';

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';

import type { TutorialLockScope, TutorialTargetRect } from '../lib/tutorial/types';

type TutorialMode = 'blocked' | 'chooser' | 'running' | 'completed';

type ChecklistItem = {
  actionId: string;
  label: string;
  details?: string;
  done: boolean;
  kind?: 'ui' | 'manual';
};

type OptionalDecisionState = {
  id: string;
  label: string;
  resolved: boolean;
  defaultLabel: string;
  actionTouched: boolean;
};

type Props = {
  open: boolean;
  mode: TutorialMode;
  isDesktop: boolean;
  hasSavedProgress: boolean;
  chapterTitle: string;
  chapterDescription: string;
  chapterIndex: number;
  chapterCount: number;
  stepTitle: string;
  stepWhat: string;
  stepImpact: string;
  stepIndex: number;
  stepCount: number;
  canGoNext: boolean;
  atStart: boolean;
  atEnd: boolean;
  checklist: ChecklistItem[];
  currentTaskOverride?: string | null;
  optionalDecision: OptionalDecisionState | null;
  targetRect: TutorialTargetRect | null;
  targetMissing: boolean;
  requiresTargetRect?: boolean;
  runningScope?: TutorialLockScope;
  onClose: () => void;
  onStartNew: () => void;
  onResume: () => void;
  onRestart: () => void;
  onBack: () => void;
  onNext: () => void;
  onFinish: () => void;
  onMarkManual: (actionId: string) => void;
  onUseOptionalDefault: (optionalDecisionId: string) => void;
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export default function TutorialOverlay({
  open,
  mode,
  isDesktop,
  hasSavedProgress,
  chapterTitle,
  chapterDescription,
  chapterIndex,
  chapterCount,
  stepTitle,
  stepWhat,
  stepImpact,
  stepIndex,
  stepCount,
  canGoNext: _canGoNext,
  atStart: _atStart,
  atEnd: _atEnd,
  checklist,
  currentTaskOverride = null,
  optionalDecision,
  targetRect,
  targetMissing,
  requiresTargetRect = false,
  runningScope = 'free',
  onClose,
  onStartNew,
  onResume,
  onRestart,
  onBack: _onBack,
  onNext: _onNext,
  onFinish: _onFinish,
  onMarkManual,
  onUseOptionalDefault,
}: Props) {
  const suppressBackdropCloseRef = useRef(false);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const mapFocusedRunning = mode === 'running' && runningScope === 'map_only';
  const sidebarFocusedRunning = mode === 'running' && runningScope === 'sidebar_section_only';
  const titleId = useId();
  const bodyId = useId();
  const [cardContentHeight, setCardContentHeight] = useState(0);
  const [targetResolveTimedOut, setTargetResolveTimedOut] = useState(false);
  const [viewport, setViewport] = useState(() => ({
    width: typeof window !== 'undefined' ? window.innerWidth : 1440,
    height: typeof window !== 'undefined' ? window.innerHeight : 900,
  }));
  const logOverlayDim = useCallback(
    (_event: string, _payload?: Record<string, unknown>) => {},
    [],
  );

  useEffect(() => {
    if (!open || mode !== 'running') return;
    const overlay = document.querySelector<HTMLElement>('.tutorialOverlay');
    const backdrop = overlay?.querySelector<HTMLElement>('.tutorialOverlay__backdrop') ?? null;
    const spotlight = overlay?.querySelector<HTMLElement>('.tutorialOverlay__spotlight') ?? null;
    const backdropStyle = backdrop ? window.getComputedStyle(backdrop) : null;
    const spotlightStyle = spotlight ? window.getComputedStyle(spotlight) : null;
    logOverlayDim('running-style-snapshot', {
      runningScope,
      mapFocusedRunning,
      sidebarFocusedRunning,
      overlayClass: overlay?.className ?? null,
      hasBackdrop: Boolean(backdrop),
      hasSpotlight: Boolean(spotlight),
      backdropBackground: backdropStyle?.backgroundColor ?? null,
      backdropDisplay: backdropStyle?.display ?? null,
      backdropOpacity: backdropStyle?.opacity ?? null,
      spotlightDisplay: spotlightStyle?.display ?? null,
      spotlightBoxShadow: spotlightStyle?.boxShadow ?? null,
    });
  }, [logOverlayDim, mapFocusedRunning, mode, open, runningScope, sidebarFocusedRunning, stepIndex]);

  useEffect(() => {
    setTargetResolveTimedOut(false);
    if (!open || mode !== 'running' || !requiresTargetRect || mapFocusedRunning || targetRect) {
      return;
    }
    const timer = window.setTimeout(() => {
      setTargetResolveTimedOut(true);
    }, 650);
    return () => window.clearTimeout(timer);
  }, [mapFocusedRunning, mode, open, requiresTargetRect, stepIndex, targetRect]);

  useEffect(() => {
    if (!open) return;
    if (typeof window === 'undefined') return;

    let raf = 0;
    const handleResize = () => {
      if (raf) window.cancelAnimationFrame(raf);
      raf = window.requestAnimationFrame(() => {
        setViewport({
          width: window.innerWidth,
          height: window.innerHeight,
        });
      });
    };

    window.addEventListener('resize', handleResize, { passive: true });
    window.addEventListener('orientationchange', handleResize);
    const vv = window.visualViewport;
    vv?.addEventListener('resize', handleResize);
    vv?.addEventListener('scroll', handleResize);
    return () => {
      if (raf) window.cancelAnimationFrame(raf);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleResize);
      vv?.removeEventListener('resize', handleResize);
      vv?.removeEventListener('scroll', handleResize);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const card = cardRef.current;
    if (!card) return;
    card.scrollTop = 0;
  }, [chapterIndex, mode, open, stepIndex]);

  useEffect(() => {
    if (!open) return;
    const node = cardRef.current;
    if (!node) return;

    const updateHeight = () => {
      setCardContentHeight(Math.ceil(node.scrollHeight));
    };

    updateHeight();

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(() => updateHeight());
      observer.observe(node);
      return () => observer.disconnect();
    }

    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, [chapterIndex, mode, open, stepIndex, checklist.length, optionalDecision?.id]);

  useEffect(() => {
    if (!open) return;
    if (mode === 'running' && runningScope === 'map_only') return;
    const card = cardRef.current;
    if (!card) return;

    const timer = window.setTimeout(() => {
      const primary = card.querySelector<HTMLElement>('.primary:not([disabled])');
      const fallback = card.querySelector<HTMLElement>('button:not([disabled])');
      (primary ?? fallback)?.focus();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [mode, open, runningScope, stepIndex]);

  useEffect(() => {
    if (!open) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose, open]);

  const layout = useMemo(() => {
    const vw = viewport.width;
    const vh = viewport.height;
    const availableWidth = Math.max(280, vw - 24);
    const cardWidth = Math.min(520, availableWidth);
    const safeViewportInset = 16;
    const maxVisibleHeight = Math.max(220, vh - safeViewportInset * 2);
    const estimatedHeight = cardContentHeight > 0 ? cardContentHeight : Math.min(vh * 0.64, 520);
    const needsScale = cardContentHeight > 0 && cardContentHeight > maxVisibleHeight;
    const scale = needsScale
      ? clamp(maxVisibleHeight / cardContentHeight, 0.84, 1)
      : 1;
    const renderedHeight = estimatedHeight * scale;
    const centeredTop = clamp(
      (vh - renderedHeight) / 2,
      safeViewportInset,
      Math.max(safeViewportInset, vh - renderedHeight - safeViewportInset),
    );

    if (
      mode === 'running' &&
      (runningScope === 'sidebar_section_only' || runningScope === 'map_only')
    ) {
      const dockedWidth = Math.min(430, availableWidth);
      return {
        cardStyle: {
          left: 12,
          top: centeredTop,
          width: dockedWidth,
          transform: scale < 1 ? `scale(${scale})` : undefined,
          transformOrigin: 'top left',
        },
        arrowStyle: null as { left: number; top: number } | null,
        arrowClass: 'isHidden',
      };
    }

    if (!targetRect || mode !== 'running') {
      return {
        cardStyle: {
          left: clamp((vw - cardWidth) / 2, 12, vw - cardWidth - 12),
          top: centeredTop,
          width: cardWidth,
          transform: scale < 1 ? `scale(${scale})` : undefined,
          transformOrigin: 'top center',
        },
        arrowStyle: null as { left: number; top: number } | null,
        arrowClass: 'isHidden',
      };
    }

    const targetRight = targetRect.left + targetRect.width;

    const preferRight = targetRight + 24 + cardWidth < vw - 16;
    const left = preferRight
      ? targetRight + 24
      : clamp(targetRect.left - cardWidth - 24, 12, vw - cardWidth - 12);

    return {
      cardStyle: {
        left,
        top: centeredTop,
        width: cardWidth,
        transform: scale < 1 ? `scale(${scale})` : undefined,
        transformOrigin: 'top left',
      },
      arrowStyle: { left, top: centeredTop },
      arrowClass: preferRight ? 'isLeftAnchor' : 'isRightAnchor',
    };
  }, [cardContentHeight, mode, runningScope, targetRect, viewport.height, viewport.width]);
  const firstPendingIndex = useMemo(
    () => checklist.findIndex((item) => !item.done),
    [checklist],
  );
  const doneCount = useMemo(
    () => checklist.filter((item) => item.done).length,
    [checklist],
  );
  const progressPercent = useMemo(() => {
    if (!checklist.length) return 100;
    return Math.round((doneCount / checklist.length) * 100);
  }, [checklist.length, doneCount]);
  const currentTask = useMemo(() => {
    if (currentTaskOverride) {
      return {
        actionId: 'override',
        label: currentTaskOverride,
        done: false,
      } as ChecklistItem;
    }
    if (!checklist.length) return null;
    return checklist.find((item) => !item.done) ?? null;
  }, [checklist, currentTaskOverride]);

  if (!open) return null;
  const waitingForTargetRect =
    mode === 'running' &&
    requiresTargetRect &&
    !mapFocusedRunning &&
    !targetRect &&
    !targetResolveTimedOut;

  function handleBackdropClick() {
    if (mode === 'running') {
      return;
    }
    if (suppressBackdropCloseRef.current) {
      suppressBackdropCloseRef.current = false;
      return;
    }
    onClose();
  }

  return (
    <div
      className={`tutorialOverlay ${mode === 'running' ? 'isRunning' : ''} ${mapFocusedRunning ? 'isMapFocused' : ''} ${sidebarFocusedRunning ? 'isSidebarFocused' : ''}`.trim()}
      role="dialog"
      aria-modal={mode === 'running' && runningScope === 'map_only' ? 'false' : 'true'}
      aria-labelledby={titleId}
      aria-describedby={bodyId}
      aria-label="Guided frontend tutorial"
    >
      {!sidebarFocusedRunning ? (
        <div className="tutorialOverlay__backdrop" onClick={handleBackdropClick} />
      ) : null}

      {mode === 'running' && targetRect && !mapFocusedRunning && !sidebarFocusedRunning ? (
        <>
          <div
            className="tutorialOverlay__spotlight"
            style={{
              top: targetRect.top - 6,
              left: targetRect.left - 6,
              width: targetRect.width + 12,
              height: targetRect.height + 12,
            }}
          />
          <div className={`tutorialOverlay__arrow ${layout.arrowClass}`} style={layout.arrowStyle ?? undefined} />
        </>
      ) : null}

      {waitingForTargetRect ? (
        <div className="tutorialOverlay__loading" aria-live="polite" aria-label="Positioning tutorial">
          <span className="tutorialOverlay__loadingSpinner" aria-hidden="true" />
        </div>
      ) : null}

      {!waitingForTargetRect ? (
        <div
          ref={cardRef}
          className="tutorialOverlay__card tutorialOverlay__card--guided"
          style={layout.cardStyle}
          onMouseDown={() => {
            suppressBackdropCloseRef.current = true;
          }}
          onMouseUp={() => {
            window.setTimeout(() => {
              suppressBackdropCloseRef.current = false;
            }, 0);
          }}
          onClick={(event) => {
            event.stopPropagation();
          }}
        >
        {mode === 'blocked' ? (
          <>
            <div className="tutorialOverlay__badge">Desktop only</div>
            <h2 id={titleId} className="tutorialOverlay__title">Tutorial requires desktop width</h2>
            <p id={bodyId} className="tutorialOverlay__body">
              This full guided tutorial is currently designed for desktop-only interaction. Expand your
              viewport to continue.
            </p>
            <div className="tutorialOverlay__actions">
              <button type="button" className="primary" onClick={onClose}>
                Close
              </button>
            </div>
          </>
        ) : null}

        {mode === 'chooser' ? (
          <>
            <div className="tutorialOverlay__badge">Guided tutorial</div>
            <h2 id={titleId} className="tutorialOverlay__title">Resume or restart tutorial</h2>
            <p id={bodyId} className="tutorialOverlay__body">
              {hasSavedProgress
                ? 'You have unfinished tutorial progress. Resume where you left off or restart from chapter one.'
                : 'Start the full guided walkthrough from chapter one.'}
            </p>
            <div className="tutorialOverlay__actions">
              {hasSavedProgress ? (
                <button type="button" className="secondary" onClick={onResume}>
                  Resume
                </button>
              ) : null}
              <button
                type="button"
                className="primary"
                onClick={hasSavedProgress ? onRestart : onStartNew}
              >
                {hasSavedProgress ? 'Restart' : 'Start tutorial'}
              </button>
            </div>
            <div className="tutorialOverlay__footer">
              <button type="button" className="ghostButton" onClick={onClose}>
                Close
              </button>
            </div>
          </>
        ) : null}

        {mode === 'completed' ? (
          <>
            <div className="tutorialOverlay__badge">Completed</div>
            <h2 id={titleId} className="tutorialOverlay__title">Tutorial completed</h2>
            <p id={bodyId} className="tutorialOverlay__body">
              You can restart the full walkthrough at any time from the Setup card.
            </p>
            <div className="tutorialOverlay__actions">
              <button type="button" className="secondary" onClick={onRestart}>
                Restart tutorial
              </button>
              <button type="button" className="primary" onClick={onClose}>
                Close
              </button>
            </div>
          </>
        ) : null}

        {mode === 'running' ? (
          <>
            <div className="tutorialOverlay__badge">
              Chapter {chapterIndex} / {chapterCount}
            </div>
            <div className="tutorialOverlay__chapterTitle">{chapterTitle}</div>
            <div className="tutorialOverlay__chapterHint">{chapterDescription}</div>
            <div className="tutorialOverlay__stepCounter">
              Step {stepIndex} / {stepCount}
            </div>
            <div className="tutorialOverlay__meterWrap" aria-hidden="true">
              <div className="tutorialOverlay__meterTrack">
                <div className="tutorialOverlay__meterFill" style={{ width: `${progressPercent}%` }} />
              </div>
              <div className="tutorialOverlay__meterMeta">
                {doneCount}/{checklist.length || 0} tasks complete
              </div>
            </div>

            <h2 id={titleId} className="tutorialOverlay__title">{stepTitle}</h2>
            <p id={bodyId} className="tutorialOverlay__body">
              <strong>What to do:</strong> {stepWhat}
            </p>
            <p className="tutorialOverlay__body">
              <strong>How results change:</strong> {stepImpact}
            </p>
            {currentTask ? (
              <div className="tutorialOverlay__currentTask" role="status" aria-live="polite">
                <span className="tutorialOverlay__currentTaskLabel">Current task</span>
                <span>{currentTask.label}</span>
              </div>
            ) : null}

            {targetMissing ? (
              <div className="warningPanel" role="status" aria-live="polite">
                Target control is not currently visible for this state. Complete available requirements
                below and continue.
              </div>
            ) : null}

            {checklist.length > 0 ? (
              <ol className="tutorialChecklist" aria-label="Step tasks in required order">
                {checklist.map((item, idx) => {
                  const previousDone = checklist.slice(0, idx).every((entry) => entry.done);
                  const isLocked = !item.done && !previousDone;
                  const isCurrent =
                    !item.done && previousDone && (firstPendingIndex === idx || firstPendingIndex === -1);
                  const stateClass = item.done
                    ? 'isDone'
                    : isCurrent
                      ? 'isCurrent'
                      : isLocked
                        ? 'isLocked'
                        : 'isPending';
                  const stateLabel = item.done
                    ? 'Done'
                    : isCurrent
                      ? 'Now'
                      : isLocked
                        ? 'Locked'
                        : 'Pending';

                  return (
                    <li
                      key={item.actionId}
                      className={`tutorialChecklist__item ${stateClass}`}
                      aria-current={isCurrent ? 'step' : undefined}
                      data-state={stateClass}
                    >
                      <span className="tutorialChecklist__number" aria-hidden="true">
                        {idx + 1}
                      </span>
                      <span className="tutorialChecklist__labelWrap">
                        <span className="tutorialChecklist__label">{item.label}</span>
                        {item.details ? (
                          <span className="tutorialChecklist__details">{item.details}</span>
                        ) : null}
                      </span>
                      <span className={`tutorialChecklist__state ${stateClass}`}>{stateLabel}</span>
                      {!item.done && item.kind === 'manual' && !isLocked ? (
                        <button
                          type="button"
                          className="ghostButton tutorialChecklist__manualBtn"
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            onMarkManual(item.actionId);
                          }}
                        >
                          {item.actionId.startsWith('map.confirm_') ? 'Confirm' : 'Mark done'}
                        </button>
                      ) : null}
                    </li>
                  );
                })}
              </ol>
            ) : null}

            {optionalDecision ? (
              <div className="tutorialOptional">
                <div className="tutorialOptional__title">Optional field decision</div>
                <div className="tutorialOptional__body">{optionalDecision.label}</div>
                {optionalDecision.resolved || optionalDecision.actionTouched ? (
                  <div className="tutorialOptional__resolved">Resolved</div>
                ) : (
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => onUseOptionalDefault(optionalDecision.id)}
                  >
                    {optionalDecision.defaultLabel}
                  </button>
                )}
              </div>
            ) : null}

            <div className="tutorialOverlay__autoAdvanceHint">
              Progression is automatic when the current task list is complete.
            </div>

            <div className="tutorialOverlay__footer">
              {!isDesktop ? null : (
                <button type="button" className="ghostButton" onClick={onClose}>
                  Close and keep progress
                </button>
              )}
            </div>
          </>
        ) : null}
        </div>
      ) : null}
    </div>
  );
}
