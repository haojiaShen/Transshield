/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TRANSSHIELD_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
