"use client"

import * as React from "react"
import dynamic from "next/dynamic"
import { Area, AreaChart, CartesianGrid, XAxis } from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group"
import { Badge } from "@/components/ui/badge"
import { Spinner } from "@/components/ui/spinner"
import { useTicker } from "@/lib/ticker-context"
import { IconTrendingUp, IconTrendingDown } from "@tabler/icons-react"

interface KpiData {
  ticker: string
  price: number
  pct_change: number
  news_volume: number
  avg_sentiment: number
  signal: string
  rsi: number
}

// Plotly loaded dynamically (SSR incompatible)
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false })

/* ── Ticker universe (mirrors config.py) ── */
const TICKER_CATEGORIES: Record<string, string[]> = {
  "🖥️ Tech": ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "AMD", "NFLX"],
  "🏦 Finance": ["JPM", "GS", "BAC", "COIN"],
  "🪙 Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
}

/* ── Colors (matches config.py) ── */
const POSITIVE_COLOR = "#00cc96"
const NEGATIVE_COLOR = "#ef553b"
const ACCENT_COLOR = "#636efa"

interface PricePoint {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  SMA_7: number
  SMA_21: number
  SMA_50: number
  BB_upper: number
  BB_lower: number
}

interface ChartPoint {
  timestamp: string
  price: number
  engouement: number
  // keep raw for candle
  open: number
  high: number
  low: number
  close: number
  volume: number
  SMA_7: number
  SMA_21: number
  SMA_50: number
  BB_upper: number
  BB_lower: number
}

/* Seeded pseudo-random for deterministic fake engouement */
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

function generateChartData(data: PricePoint[]): ChartPoint[] {
  if (data.length === 0) return []
  const rng = seededRandom(42)

  // Engouement is a synthetic "market enthusiasm" curve that follows price
  // with noise + occasional divergences (simulating prediction market signal)
  const closes = data.map((d) => d.close)
  const minP = Math.min(...closes)
  const maxP = Math.max(...closes)
  const range = maxP - minP || 1

  const raw = data.map((d, i) => {
    const noise = (rng() - 0.5) * range * 0.25
    const drift = Math.sin(i * 0.7) * range * 0.08
    const spike = rng() > 0.85 ? (rng() - 0.5) * range * 0.4 : 0
    return d.close + noise + drift + spike
  })
  // Light 3-point smoothing
  const smoothed = raw.map((v, i) => {
    const start = Math.max(0, i - 1)
    const end = Math.min(raw.length - 1, i + 1)
    let sum = 0
    let count = 0
    for (let j = start; j <= end; j++) {
      sum += raw[j]
      count++
    }
    return Math.round((sum / count) * 100) / 100
  })

  return data.map((d, i) => ({
    ...d,
    price: d.close,
    engouement: Math.round(smoothed[i] * 100) / 100,
  }))
}

const areaConfig = {
  price: {
    label: "Price ($)",
    color: "hsl(0 0% 100%)",
  },
  engouement: {
    label: "Engouement",
    color: "hsl(0 0% 65%)",
  },
} satisfies ChartConfig

const periodLabels: Record<string, string> = {
  "7d": "7 days",
  "1mo": "1 month",
  "3mo": "3 months",
  "6mo": "6 months",
  "1y": "1 year",
}

export function ChartAreaInteractive() {
  const [period, setPeriod] = React.useState("3mo")
  const { ticker, setTicker } = useTicker()
  const [chartMode, setChartMode] = React.useState<"area" | "candle" | "indicators">("candle")
  const [rawData, setRawData] = React.useState<PricePoint[]>([])
  const [chartData, setChartData] = React.useState<ChartPoint[]>([])
  const [loading, setLoading] = React.useState(true)
  const [kpiData, setKpiData] = React.useState<KpiData | null>(null)

  React.useEffect(() => {
    setLoading(true)
    fetch(
      `http://localhost:8000/api/price-history?ticker=${ticker}&period=${period}`
    )
      .then((r) => r.json())
      .then((d: PricePoint[]) => {
        setRawData(d)
        setChartData(generateChartData(d))
      })
      .catch(() => {
        setRawData([])
        setChartData([])
      })
      .finally(() => setLoading(false))
  }, [period, ticker])

  // Fetch KPI data for the indicators tab
  React.useEffect(() => {
    fetch(`http://localhost:8000/api/kpis?ticker=${ticker}`)
      .then((r) => r.json())
      .then((d: KpiData) => setKpiData(d))
      .catch(() => setKpiData(null))
  }, [ticker])

  /* ── Plotly candlestick (exact replica of visualizer.py) ── */
  const renderCandlestick = () => {
    const timestamps = rawData.map((d) => d.timestamp)
    const opens = rawData.map((d) => d.open)
    const highs = rawData.map((d) => d.high)
    const lows = rawData.map((d) => d.low)
    const closes = rawData.map((d) => d.close)
    const volumes = rawData.map((d) => d.volume)
    const volColors = rawData.map((d) =>
      d.close >= d.open ? POSITIVE_COLOR : NEGATIVE_COLOR
    )

    // SMA overlays
    const smaTraces = (["SMA_7", "SMA_21", "SMA_50"] as const)
      .filter((key) => rawData.some((d) => d[key] && d[key] !== 0))
      .map((key) => ({
        x: timestamps,
        y: rawData.map((d) => d[key] || null),
        mode: "lines" as const,
        name: key,
        line: { width: 1, dash: "dot" as const },
        xaxis: "x" as const,
        yaxis: "y" as const,
      }))

    // Bollinger bands
    const bbTraces =
      rawData.some((d) => d.BB_upper && d.BB_upper !== 0)
        ? [
            {
              x: timestamps,
              y: rawData.map((d) => d.BB_upper || null),
              mode: "lines" as const,
              name: "BB Upper",
              line: { width: 0.5, color: "rgba(99,110,250,0.3)" },
              showlegend: false,
              xaxis: "x" as const,
              yaxis: "y" as const,
            },
            {
              x: timestamps,
              y: rawData.map((d) => d.BB_lower || null),
              mode: "lines" as const,
              name: "BB Lower",
              fill: "tonexty" as const,
              line: { width: 0.5, color: "rgba(99,110,250,0.3)" },
              fillcolor: "rgba(99,110,250,0.07)",
              showlegend: false,
              xaxis: "x" as const,
              yaxis: "y" as const,
            },
          ]
        : []

    return (
      <Plot
        data={[
          // Candlestick
          {
            type: "candlestick",
            x: timestamps,
            open: opens,
            high: highs,
            low: lows,
            close: closes,
            name: "OHLC",
            increasing: { line: { color: POSITIVE_COLOR } },
            decreasing: { line: { color: NEGATIVE_COLOR } },
            xaxis: "x",
            yaxis: "y",
          },
          // SMA overlays
          ...smaTraces,
          // Bollinger bands
          ...bbTraces,
          // Volume bars (subplot)
          {
            type: "bar",
            x: timestamps,
            y: volumes,
            name: "Volume",
            marker: { color: volColors, opacity: 0.5 },
            xaxis: "x",
            yaxis: "y2",
          },
        ]}
        layout={{
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          template: "plotly_dark" as any,
          height: 440,
          margin: { l: 50, r: 20, t: 30, b: 30 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#94a3b8", size: 11 },
          legend: {
            orientation: "h",
            y: 1.08,
            x: 0.5,
            xanchor: "center",
            font: { size: 10, color: "#94a3b8" },
          },
          xaxis: {
            rangeslider: { visible: false },
            gridcolor: "rgba(148,163,184,0.08)",
            type: "date",
          },
          yaxis: {
            domain: [0.28, 1],
            gridcolor: "rgba(148,163,184,0.08)",
            tickprefix: "$",
            side: "right",
          },
          yaxis2: {
            domain: [0, 0.22],
            gridcolor: "rgba(148,163,184,0.08)",
            showticklabels: false,
          },
          hovermode: "x unified",
        }}
        config={{
          displayModeBar: false,
          responsive: true,
        }}
        useResizeHandler
        style={{ width: "100%", height: "440px" }}
      />
    )
  }

  return (
    <Card className="@container/card pt-0">
      <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
        <div className="grid flex-1 gap-1">
          <div className="flex items-center gap-2">
            <CardTitle>{ticker} — Price</CardTitle>
            <Select value={ticker} onValueChange={setTicker}>
              <SelectTrigger
                className="w-[130px] text-sm"
                aria-label="Select ticker"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="rounded-xl">
                <ScrollArea className="h-[280px]">
                  {Object.entries(TICKER_CATEGORIES).map(([cat, tickers]) => (
                    <React.Fragment key={cat}>
                      <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                        {cat}
                      </div>
                      {tickers.map((t) => (
                        <SelectItem key={t} value={t} className="rounded-lg pl-4">
                          {t}
                        </SelectItem>
                      ))}
                    </React.Fragment>
                  ))}
                </ScrollArea>
              </SelectContent>
            </Select>
          </div>
          <CardDescription>
            {chartMode === "area"
              ? `Price × Engouement — ${periodLabels[period]}`
              : chartMode === "candle"
                ? `OHLC Candlestick + Volume — ${periodLabels[period]}`
                : `Financial Indicators — ${ticker}`}
          </CardDescription>
        </div>
        <div className="flex items-center gap-2">
          {/* Chart mode toggle */}
          <ToggleGroup
            type="single"
            value={chartMode}
            onValueChange={(v) => v && setChartMode(v as "area" | "candle" | "indicators")}
            variant="outline"
            className="*:data-[slot=toggle-group-item]:!px-3"
          >
            <ToggleGroupItem value="area" aria-label="Area chart">
              Area
            </ToggleGroupItem>
            <ToggleGroupItem value="candle" aria-label="Candlestick chart">
              Candle
            </ToggleGroupItem>
            <ToggleGroupItem value="indicators" aria-label="Financial indicators">
              Indicators
            </ToggleGroupItem>
          </ToggleGroup>
          {/* Period selector */}
          <Select value={period} onValueChange={setPeriod}>
            <SelectTrigger
              className="hidden w-[140px] rounded-lg sm:ml-auto sm:flex"
              aria-label="Select period"
            >
              <SelectValue placeholder="3 months" />
            </SelectTrigger>
            <SelectContent className="rounded-xl">
              {Object.entries(periodLabels).map(([val, label]) => (
                <SelectItem key={val} value={val} className="rounded-lg">
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        {loading && (
          <div className="flex h-[250px] items-center justify-center gap-2">
            <Spinner className="size-5" />
            <span className="text-sm text-muted-foreground">
              Loading price data…
            </span>
          </div>
        )}
        {!loading && chartData.length === 0 && (
          <div className="flex h-[250px] items-center justify-center text-sm text-muted-foreground">
            No price data available
          </div>
        )}
        {!loading && chartData.length > 0 && chartMode === "candle" &&
          renderCandlestick()
        }
        {!loading && chartData.length > 0 && chartMode === "area" && (
          <ChartContainer
            config={areaConfig}
            className="aspect-auto h-[250px] w-full"
          >
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="fillPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop
                    offset="5%"
                    stopColor="var(--color-price)"
                    stopOpacity={0.8}
                  />
                  <stop
                    offset="95%"
                    stopColor="var(--color-price)"
                    stopOpacity={0.1}
                  />
                </linearGradient>
                <linearGradient id="fillEngouement" x1="0" y1="0" x2="0" y2="1">
                  <stop
                    offset="5%"
                    stopColor="var(--color-engouement)"
                    stopOpacity={0.8}
                  />
                  <stop
                    offset="95%"
                    stopColor="var(--color-engouement)"
                    stopOpacity={0.1}
                  />
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="timestamp"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
                tickFormatter={(value) => {
                  const date = new Date(value)
                  return date.toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                  })
                }}
              />
              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent
                    labelFormatter={(value) => {
                      return new Date(value).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                      })
                    }}
                    indicator="dot"
                  />
                }
              />
              <Area
                dataKey="engouement"
                type="natural"
                fill="url(#fillEngouement)"
                stroke="var(--color-engouement)"
              />
              <Area
                dataKey="price"
                type="natural"
                fill="url(#fillPrice)"
                stroke="var(--color-price)"
              />
              <ChartLegend content={<ChartLegendContent />} />
            </AreaChart>
          </ChartContainer>
        )}
        {/* ── Financial Indicators tab ── */}
        {!loading && chartMode === "indicators" && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {/* RSI Card */}
            {kpiData ? (() => {
              const rsi = kpiData.rsi
              const rsiStatus = rsi > 70 ? "Overbought" : rsi < 30 ? "Oversold" : "Neutral"
              const rsiColor = rsi > 70 ? "#ef4444" : rsi < 30 ? "#22c55e" : "#f59e0b"
              const barPct = Math.min(100, Math.max(0, rsi))
              return (
                <div className="rounded-xl border bg-card p-5 shadow-xs">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium text-muted-foreground">RSI (14)</span>
                    <Badge variant="outline" style={{ color: rsiColor, borderColor: rsiColor }}>
                      {rsi > 50 ? <IconTrendingUp className="mr-1 size-3" /> : <IconTrendingDown className="mr-1 size-3" />}
                      {rsiStatus}
                    </Badge>
                  </div>
                  <div className="text-3xl font-bold tabular-nums" style={{ color: rsiColor }}>
                    {rsi.toFixed(1)}
                  </div>
                  {/* RSI bar */}
                  <div className="mt-3 relative h-3 w-full rounded-full bg-muted overflow-hidden">
                    {/* Zone markers */}
                    <div className="absolute inset-0 flex">
                      <div className="w-[30%] bg-green-500/15" />
                      <div className="w-[40%] bg-yellow-500/15" />
                      <div className="w-[30%] bg-red-500/15" />
                    </div>
                    {/* Current value indicator */}
                    <div
                      className="absolute top-0 h-full w-1 rounded-full"
                      style={{ left: `${barPct}%`, backgroundColor: rsiColor, transform: "translateX(-50%)" }}
                    />
                  </div>
                  <div className="mt-1.5 flex justify-between text-[10px] text-muted-foreground">
                    <span>0 — Oversold</span>
                    <span>50</span>
                    <span>Overbought — 100</span>
                  </div>
                  <p className="mt-3 text-xs text-muted-foreground">
                    14-period Relative Strength Index
                  </p>
                </div>
              )
            })() : (
              <div className="flex h-[200px] items-center justify-center rounded-xl border bg-card">
                <Spinner className="size-5" />
              </div>
            )}

            {/* Placeholder for future indicators */}
            <div className="flex h-full min-h-[200px] items-center justify-center rounded-xl border border-dashed bg-card/50 text-sm text-muted-foreground">
              More indicators coming soon…
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
