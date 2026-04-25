export interface ProviderConfig {
  model: string;
  base_url: string;
  api_key: string;
  context_window?: number;
}

const STORAGE_KEY = 'ml-intern-openai-provider';

export function loadProviderConfig(): ProviderConfig | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw) as ProviderConfig;
    if (!data.model || !data.base_url || !data.api_key) return null;
    return data;
  } catch {
    return null;
  }
}

export function saveProviderConfig(config: ProviderConfig): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

export function providerRequestBody(extra?: Record<string, unknown>): string {
  const provider = loadProviderConfig();
  return JSON.stringify({
    ...(extra || {}),
    ...(provider ? { provider } : {}),
  });
}
