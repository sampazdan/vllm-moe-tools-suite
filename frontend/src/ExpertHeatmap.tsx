import { HeatmapChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef, useState } from "react";

import type { ExpertProfile, RoutingSummary } from "./types";

interface ExpertHeatmapProps {
  summary: RoutingSummary;
  metric: "routing_mass" | "selection_counts";
  comparisonValues?: number[][] | null;
  mode?: "references" | "selection" | "delta";
  profile?: ExpertProfile | null;
  onExpertClick?: (layerId: number, expertId: number) => void;
}

echarts.use([
  HeatmapChart,
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

export function ExpertHeatmap({
  summary,
  metric,
  comparisonValues,
  mode = "references",
  profile,
  onExpertClick,
}: ExpertHeatmapProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [visibilityRevision, setVisibilityRevision] = useState(0);

  useEffect(() => {
    if (!rootRef.current) return;
    if (rootRef.current.clientWidth === 0 || rootRef.current.clientHeight === 0) {
      const visibilityObserver = new ResizeObserver(() => {
        if (!rootRef.current?.clientWidth || !rootRef.current.clientHeight) return;
        visibilityObserver.disconnect();
        setVisibilityRevision((revision) => revision + 1);
      });
      visibilityObserver.observe(rootRef.current);
      return () => visibilityObserver.disconnect();
    }
    const chart = echarts.init(rootRef.current, undefined, { renderer: "canvas" });
    const sourceValues = summary[metric];
    const values = mode === "delta" && comparisonValues
      ? sourceValues.map((experts, layerIndex) =>
          experts.map(
            (value, expertIndex) =>
              value - (comparisonValues[layerIndex]?.[expertIndex] ?? 0),
          ),
        )
      : sourceValues;
    const data: Array<
      [number, number, number] | {
        value: [number, number, number];
        itemStyle: { borderColor: string; borderWidth: number; opacity?: number };
      }
    > = [];
    values.forEach((experts, layerIndex) => {
      experts.forEach((value, expertIndex) => {
        if (mode !== "selection" || !profile) {
          data.push([expertIndex, layerIndex, value]);
          return;
        }
        const layerId = summary.layer_ids[layerIndex];
        const configured = profile.layers[String(layerId)];
        const selected = !configured || configured.keep.includes(expertIndex);
        data.push({
          value: [expertIndex, layerIndex, value],
          itemStyle: {
            borderColor: selected ? "#344d30" : "#d4ccc0",
            borderWidth: selected ? 1.25 : 0.25,
            opacity: selected ? 1 : 0.24,
          },
        });
      });
    });
    const flatValues = values.flat();
    const maxValue = Math.max(
      ...flatValues.map((value) => Math.abs(value)),
      Number.EPSILON,
    );
    chart.setOption({
      animation: false,
      grid: { left: 52, right: 22, top: 12, bottom: 44 },
      tooltip: {
        position: "top",
        formatter: (params: { value: [number, number, number] }) => {
          const [expert, layer, value] = params.value;
          const label = metric === "routing_mass" ? value.toFixed(4) : value;
          const prefix = mode === "delta" && value > 0 ? "+" : "";
          return `Layer ${summary.layer_ids[layer]} · Expert ${expert}<br/>${prefix}${label}`;
        },
      },
      xAxis: {
        type: "category",
        data: Array.from({ length: values[0]?.length ?? 0 }, (_, index) => index),
        name: "Expert",
        nameLocation: "middle",
        nameGap: 28,
        axisLabel: { interval: 31, color: "#746f65", fontSize: 10 },
        axisLine: { lineStyle: { color: "#d7d0c3" } },
        axisTick: { show: false },
      },
      yAxis: {
        type: "category",
        data: summary.layer_ids,
        name: "Layer",
        nameLocation: "middle",
        nameGap: 36,
        axisLabel: { interval: 4, color: "#746f65", fontSize: 10 },
        axisLine: { lineStyle: { color: "#d7d0c3" } },
        axisTick: { show: false },
      },
      visualMap: {
        min: mode === "delta" ? -maxValue : 0,
        max: maxValue,
        show: false,
        inRange: {
          color: mode === "delta"
            ? ["#8d3e34", "#eadfd6", "#eef0e8", "#596f52"]
            : ["#f1ede4", "#d9b88f", "#b86843", "#6e2f26"],
        },
      },
      series: [
        {
          type: "heatmap",
          data,
          progressive: 2500,
          emphasis: {
            itemStyle: { borderColor: "#241f1a", borderWidth: 1 },
          },
        },
      ],
    });
    if (onExpertClick) {
      chart.on("click", (parameters: unknown) => {
        const value = (parameters as { value?: [number, number, number] }).value;
        if (!value) return;
        const [expertId, layerIndex] = value;
        onExpertClick(summary.layer_ids[layerIndex], expertId);
      });
    }
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(rootRef.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [comparisonValues, metric, mode, onExpertClick, profile, summary, visibilityRevision]);

  return <div className="heatmap" ref={rootRef} aria-label="Expert engagement heatmap" />;
}
