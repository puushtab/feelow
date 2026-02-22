"use client"

import * as React from "react"

interface GaugeProps {
  value: number
  size?: number
  className?: string
}

function valueToColor(v: number): string {
  const t = (v + 1) / 2
  if (t <= 0.5) {
    const r = Math.round(239 + (245 - 239) * (t / 0.5))
    const g = Math.round(68 + (158 - 68) * (t / 0.5))
    const b = Math.round(68 + (11 - 68) * (t / 0.5))
    return `rgb(${r},${g},${b})`
  }
  const r = Math.round(245 + (34 - 245) * ((t - 0.5) / 0.5))
  const g = Math.round(158 + (197 - 158) * ((t - 0.5) / 0.5))
  const b = Math.round(11 + (94 - 11) * ((t - 0.5) / 0.5))
  return `rgb(${r},${g},${b})`
}

function valueToLabel(v: number): string {
  if (v >= 0.5) return "Strong Buy"
  if (v >= 0.15) return "Buy"
  if (v > -0.15) return "Hold"
  if (v > -0.5) return "Sell"
  return "Strong Sell"
}

export function Gauge({ value, size = 160, className }: GaugeProps) {
  const clamped = Math.max(-1, Math.min(1, value))
  const color = valueToColor(clamped)
  const label = valueToLabel(clamped)

  const t = (clamped + 1) / 2 // [0, 1]
  const sw = size * 0.07        // stroke width (thin arc)
  const r = (size - sw) / 2     // radius fits inside size
  const cx = size / 2
  const cy = size / 2           // center at middle; arc drawn upward
  const half = Math.PI * r      // semicircle length

  // Upper semicircle: M left A r r 0 0 0 right  (sweep=0 → counterclockwise = upward)
  const arcD = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`

  return (
    <div className={className} style={{ width: size, textAlign: "center" }}>
      <svg width={size} height={size / 2 + sw / 2 + 2} overflow="visible"
           viewBox={`0 ${cy - r - sw / 2} ${size} ${r + sw / 2 + 2}`}>
        <defs>
          <linearGradient id={`gg-${size}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#ef4444" />
            <stop offset="50%"  stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#22c55e" />
          </linearGradient>
        </defs>

        {/* Muted full-track */}
        <path d={arcD} fill="none" stroke="hsl(var(--muted))"
              strokeWidth={sw} strokeLinecap="round" />

        {/* Colored active portion via dasharray */}
        <path d={arcD} fill="none" stroke={`url(#gg-${size})`}
              strokeWidth={sw} strokeLinecap="round"
              strokeDasharray={`${t * half} ${half}`} />
      </svg>

      <div style={{ marginTop: 6, lineHeight: 1.25 }}>
        <div style={{ fontSize: size * 0.18, fontWeight: 700, color, fontVariantNumeric: "tabular-nums" }}>
          {clamped >= 0 ? "+" : ""}{clamped.toFixed(2)}
        </div>
        <div style={{ fontSize: size * 0.09, fontWeight: 500, color: "hsl(var(--muted-foreground))" }}>
          {label}
        </div>
      </div>
    </div>
  )
}

export { valueToColor, valueToLabel }
