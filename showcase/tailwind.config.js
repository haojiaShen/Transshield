/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#111827",
        panel: "#ffffff",
        line: "rgba(15, 23, 42, 0.08)",
        accent: {
          DEFAULT: "#0f766e",
          soft: "#ccfbf1",
          strong: "#115e59"
        }
      },
      boxShadow: {
        diffusion: "0 22px 50px -28px rgba(15, 23, 42, 0.18)"
      },
      fontFamily: {
        sans: ["Outfit", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"]
      }
    }
  },
  plugins: []
}
