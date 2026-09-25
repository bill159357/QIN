import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

const projectRoot = fileURLToPath(new URL('.', import.meta.url))
const backdropDir = path.resolve(projectRoot, 'public', 'backgroud')
const backdropImageUrls = fs.existsSync(backdropDir)
  ? fs
      .readdirSync(backdropDir)
      .filter((name) => /\.(png|jpe?g|webp|avif|gif)$/i.test(name))
      .sort((a, b) => a.localeCompare(b))
      .map((name) => `/backgroud/${name}`)
  : []

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    'import.meta.env.VITE_JIANDU_BACKDROP_IMAGES': JSON.stringify(backdropImageUrls),
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
