import { HeatmapChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";

import type { RoutingSummary } from "./types";

interface ExpertHeatmapProps {
  summary: RoutingSummary;
  metric: "routing_mass" | "selection_counts";
}

echarts.use([
  HeatmapChart,
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

export function ExpertHeatmap({ summary, metric }: ExpertHeatmapProps) {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!rootRef.current) return;
    const chart = echarts.init(rootRef.current, undefined, { renderer: "canvas" });
    const values = summary[metric];
    const data: [number, number, number][] = [];
    values.forEach((experts, layerIndex) => {
      experts.forEach((value, expertIndex) => {
        data.push([expertIndex, layerIndex, value]);
      });
    });
    const maxValue = Math.max(...data.map((entry) => entry[2]), 1);
    chart.setOption({
      animation: false,
      grid: { left: 52, right: 22, top: 12, bottom: 44 },
      tooltip: {
        position: "top",
        formatter: (params: { value: [number, number, number] }) => {
          const [expert, layer, value] = params.value;
          const label = metric === "routing_mass" ? value.toFixed(3) : value;
          return `Layer ${summary.layer_ids[layer]} · Expert ${expert}<br/>${label}`;
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
        min: 0,
        max: maxValue,
        show: false,
        inRange: {
          color: ["#f1ede4", "#d9b88f", "#b86843", "#6e2f26"],
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
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(rootRef.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [metric, summary]);

  return <div className="heatmap" ref={rootRef} aria-label="Expert engagement heatmap" />;
}
