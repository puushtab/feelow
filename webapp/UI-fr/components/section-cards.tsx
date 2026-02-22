"use client"

import * as React from "react"
import { IconTrendingDown, IconTrendingUp } from "@tabler/icons-react"

import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Spinner } from '@/components/ui/spinner'
import { Gauge, valueToColor, valueToLabel } from '@/components/ui/gauge'
import { useTicker, type PolymarketData } from '@/lib/ticker-context'

interface KpiData {
  ticker: string
  price: number
  pct_change: number
  news_volume: number
  avg_sentiment: number
  signal: string
  rsi: number
}

export function SectionCards() {
  const { ticker, polymarket } = useTicker()
  const [data, setData] = React.useState<KpiData | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    setLoading(true)
    fetch(`http://localhost:8000/api/kpis?ticker=${ticker}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d: KpiData) => {
        setData(d)
        setError(null)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [ticker])

  if (loading) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 px-4">
        <Spinner className="size-5" />
        <span className="text-sm text-muted-foreground">Loading KPIs…</span>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="px-4 lg:px-6 text-sm text-destructive">
        API error: {error ?? "No data"} — is the backend running on :8000?
      </div>
    )
  }

  const priceUp = data.pct_change >= 0
  const sentimentUp = data.avg_sentiment >= 0
  const rsiStatus =
    data.rsi > 70 ? "Overbought" : data.rsi < 30 ? "Oversold" : "Neutral zone"

  const signalColor =
    data.signal.includes("BUY")
      ? "text-green-500"
      : data.signal.includes("SELL")
        ? "text-red-500"
        : "text-yellow-500"

  return (
    <div className="*:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card dark:*:data-[slot=card]:bg-card grid grid-cols-1 gap-4 px-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:shadow-xs lg:px-6 @xl/main:grid-cols-2 @5xl/main:grid-cols-4">
      {/* Price */}
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>Price ({data.ticker})</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            ${data.price.toLocaleString("en-US", { minimumFractionDigits: 2 })}
          </CardTitle>
          <CardAction>
            <Badge variant="outline">
              {priceUp ? <IconTrendingUp /> : <IconTrendingDown />}
              {data.pct_change >= 0 ? "+" : ""}{data.pct_change}%
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            {priceUp ? "Trending up" : "Trending down"} this week
            {priceUp ? (
              <IconTrendingUp className="size-4" />
            ) : (
              <IconTrendingDown className="size-4" />
            )}
          </div>
          <div className="text-muted-foreground">7-day price change</div>
        </CardFooter>
      </Card>



      {/* Avg Sentiment */}
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>Avg Sentiment</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {data.avg_sentiment >= 0 ? "+" : ""}{data.avg_sentiment.toFixed(3)}
          </CardTitle>
          <CardAction>
            <Badge variant="outline" className={signalColor}>
              {sentimentUp ? <IconTrendingUp /> : <IconTrendingDown />}
              {data.signal}
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            AI Signal: <span className={signalColor}>{data.signal}</span>
          </div>
          <div className="text-muted-foreground">
            FinBERT sentiment analysis
          </div>
        </CardFooter>
      </Card>

      {/* Polymarket Engouement */}
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>Polymarket Score</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {polymarket.loading
              ? "…"
              : polymarket.error
                ? "N/A"
                : (polymarket.global_score * 100).toFixed(1) + "%"}
          </CardTitle>
          <CardAction>
            <Badge variant="outline">
              {polymarket.markets.length > 0 ? <IconTrendingUp /> : <IconTrendingDown />}
              {polymarket.markets.length} bet{polymarket.markets.length !== 1 ? "s" : ""}
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            {polymarket.loading
              ? "Loading…"
              : polymarket.markets.length > 0
                ? `Top: ${polymarket.markets[0]?.probability?.toFixed(0)}% YES`
                : "No active bets"}
          </div>
          <div className="text-muted-foreground">
            Live Polymarket prediction markets
          </div>
        </CardFooter>
      </Card>

      {/* Combined Gauge — mean of sentiment + polymarket */}
      {(() => {
        const sentiment = data.avg_sentiment ?? 0 // already in [-1, 1]
        const polyScore = polymarket.loading || polymarket.error
          ? 0
          : (polymarket.global_score ?? 0) * 2 - 1 // map [0,1] → [-1,1]
        const combined = (sentiment + polyScore) / 2
        const color = valueToColor(combined)
        const label = valueToLabel(combined)
        return (
          <Card className="@container/card">
            <CardHeader>
              <CardDescription>Investment Signal</CardDescription>
              <CardAction>
                <Badge variant="outline" style={{ color, borderColor: color }}>
                  {combined >= 0 ? <IconTrendingUp /> : <IconTrendingDown />}
                  {label}
                </Badge>
              </CardAction>
            </CardHeader>
            <div className="flex items-center justify-center pb-2">
              <Gauge value={combined} size={150} />
            </div>
            <CardFooter className="flex-col items-start gap-1.5 text-sm">
              <div className="line-clamp-1 flex gap-2 font-medium">
                Sentiment {sentiment >= 0 ? "+" : ""}{sentiment.toFixed(2)} · Poly {((polymarket.global_score ?? 0) * 100).toFixed(0)}%
              </div>
              <div className="text-muted-foreground">
                Mean of FinBERT + Polymarket scores
              </div>
            </CardFooter>
          </Card>
        )
      })()}
    </div>
  )
}
