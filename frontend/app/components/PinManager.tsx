'use client';

import { useMemo } from 'react';

import type { PinDisplayNode, PinSelectionId } from '../lib/types';

type Props = {
  nodes: PinDisplayNode[];
  selectedPinId: PinSelectionId | null;
  disabled: boolean;
  hasStop: boolean;
  canAddStop: boolean;
  oneStopHint?: string | null;
  onSelectPin: (id: PinSelectionId) => void;
  onRenameStop: (name: string) => void;
  onAddStop: () => void;
  onDeleteStop: () => void;
  onSwapPins: () => void;
  onClearPins: () => void;
};

function fmtCoord(value: number): string {
  return Number.isFinite(value) ? value.toFixed(5) : String(value);
}

export default function PinManager({
  nodes,
  selectedPinId,
  disabled,
  hasStop,
  canAddStop,
  oneStopHint = null,
  onSelectPin,
  onRenameStop,
  onAddStop,
  onDeleteStop,
  onSwapPins,
  onClearPins,
}: Props) {
  const originNode = useMemo(() => nodes.find((node) => node.id === 'origin') ?? null, [nodes]);
  const stopNode = useMemo(() => nodes.find((node) => node.id === 'stop-1') ?? null, [nodes]);
  const destinationNode = useMemo(
    () => nodes.find((node) => node.id === 'destination') ?? null,
    [nodes],
  );
  const railNodes = useMemo(
    () => [originNode, stopNode, destinationNode].filter((node): node is PinDisplayNode => Boolean(node)),
    [originNode, stopNode, destinationNode],
  );

  function renderNodeRow(
    node: PinDisplayNode,
    label: string,
    options: {
      editable: boolean;
      onRename?: (name: string) => void;
      description: string;
    },
  ) {
    const isSelected = selectedPinId === node.id;
    return (
      <li key={node.id} className={`pinManager__row ${isSelected ? 'isSelected' : ''}`}>
        <button
          type="button"
          className="pinManager__nodeBtn"
          onClick={() => onSelectPin(node.id)}
          disabled={disabled}
          aria-pressed={isSelected}
          aria-label={`${isSelected ? 'Deselect' : 'Select'} ${label} pin`}
          data-tutorial-action="pins.sidebar_select"
        >
          <span className={`pinManager__dot pinManager__dot--${node.kind}`} style={{ background: node.color }} />
          <span className="pinManager__nodeText">
            <span className="pinManager__nodeTitle">{label}</span>
            <span className="pinManager__nodeCoords">
              {fmtCoord(node.lat)}, {fmtCoord(node.lon)}
            </span>
            <span className="pinManager__nodeHint">{options.description}</span>
          </span>
        </button>

        <div className="pinManager__inlineActions">
          {options.editable ? (
            <input
              className="input pinManager__nameInput"
              value={node.label}
              onChange={(event) => options.onRename?.(event.target.value)}
              placeholder={node.label || 'Stop #1'}
              disabled={disabled}
              aria-label={`${label} name`}
            />
          ) : (
            <div className="pinManager__lockedTag" aria-label={`${label} name is fixed`}>
              <span className="pinManager__lockedDot" aria-hidden="true">•</span>
              <span>Locked</span>
            </div>
          )}
        </div>
      </li>
    );
  }

  return (
    <section className="card pinManager" data-tutorial-id="pins.section">
      <div className="sectionTitle">Pins & stops</div>
      <div className="sectionHint">Start/End are fixed labels. Stop #1 can be renamed and moved.</div>

      <div className="pinManager__rail" role="list" aria-label="Pin route order">
        {railNodes.length === 0 ? (
          <span className="pinManager__railEmpty">Add Start and End pins to build a route.</span>
        ) : (
          railNodes.map((node, idx) => {
            const isSelected = selectedPinId === node.id;
            return (
              <div key={node.id} className="pinManager__railItem" role="listitem">
                <button
                  type="button"
                  className={`pinManager__railNode ${isSelected ? 'isSelected' : ''}`}
                  style={{ color: node.color }}
                  disabled={disabled}
                  onClick={() => onSelectPin(node.id)}
                  aria-pressed={isSelected}
                  aria-label={`${isSelected ? 'Deselect' : 'Select'} ${node.label}`}
                >
                  <span className="pinManager__railNodeText">{node.label}</span>
                  {isSelected ? <span className="pinManager__railSelected">Selected</span> : null}
                </button>
                {idx < railNodes.length - 1 ? (
                  <span className="pinManager__arrow" aria-hidden="true">→</span>
                ) : null}
              </div>
            );
          })
        )}
      </div>

      <ul className="pinManager__list">
        {originNode
          ? renderNodeRow(originNode, 'Start', {
              editable: false,
              description: 'Route start point.',
            })
          : null}
        {stopNode
          ? renderNodeRow(stopNode, stopNode.label || 'Stop #1', {
              editable: true,
              onRename: onRenameStop,
              description: 'Optional intermediate stop. Name is editable.',
            })
          : null}
        {destinationNode
          ? renderNodeRow(destinationNode, 'End', {
              editable: false,
              description: 'Route end point.',
            })
          : null}
      </ul>

      <div className="pinManager__actions">
        <button
          type="button"
          className="secondary"
          disabled={disabled || !canAddStop}
          onClick={onAddStop}
          title={!canAddStop ? 'Set both Start and End to add a stop.' : 'Add midpoint stop'}
          data-tutorial-action="pins.add_stop"
        >
          {hasStop ? 'Replace stop' : 'Add stop'}
        </button>
        <button
          type="button"
          className="secondary"
          disabled={disabled || !hasStop}
          onClick={onDeleteStop}
          title={!hasStop ? 'No stop exists to delete.' : 'Delete stop'}
          data-tutorial-action="pins.delete_stop"
        >
          Delete stop
        </button>
        <button type="button" className="secondary" disabled={disabled} onClick={onSwapPins}>
          Swap start/end
        </button>
        <button type="button" className="secondary" disabled={disabled} onClick={onClearPins}>
          Clear pins
        </button>
      </div>

      <div className="pinManager__reorder">
        <button
          type="button"
          className="secondary"
          disabled={true}
          title="Unavailable in one-stop mode"
          aria-disabled="true"
        >
          Move stop up
        </button>
        <button
          type="button"
          className="secondary"
          disabled={true}
          title="Unavailable in one-stop mode"
          aria-disabled="true"
        >
          Move stop down
        </button>
      </div>
      <div className="tiny">Reorder is disabled in one-stop mode. Enable when multi-stop support is added.</div>
      <div className="tiny">Tip: click into Stop #1 name to rename, then drag the stop marker on the map.</div>
      {oneStopHint ? <div className="pinManager__hintError">{oneStopHint}</div> : null}
    </section>
  );
}
