export type MetricTooltip = {
  definition: string;
  direction: string;
  unit: string;
  note?: string;
};

export function formatMetricTooltip(tooltip: MetricTooltip): string {
  return [
    `Definition: ${tooltip.definition}`,
    `Direction: ${tooltip.direction}`,
    `Unit: ${tooltip.unit}`,
    tooltip.note ? `Note: ${tooltip.note}` : null,
  ]
    .filter(Boolean)
    .join(' ');
}
