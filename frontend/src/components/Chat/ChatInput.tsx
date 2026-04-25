import { useState, useCallback, useEffect, useRef, KeyboardEvent } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  TextField,
  Typography,
} from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import HubOutlinedIcon from '@mui/icons-material/HubOutlined';
import SettingsOutlinedIcon from '@mui/icons-material/SettingsOutlined';
import StopIcon from '@mui/icons-material/Stop';
import { apiFetch } from '@/utils/api';
import {
  loadProviderConfig,
  saveProviderConfig,
  type ProviderConfig,
} from '@/utils/provider';

interface ChatInputProps {
  sessionId?: string;
  onSend: (text: string) => void;
  onStop?: () => void;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

const defaultProvider = (): ProviderConfig => (
  loadProviderConfig() || {
    model: '',
    base_url: '',
    api_key: '',
    context_window: 200000,
  }
);

export default function ChatInput({
  sessionId,
  onSend,
  onStop,
  isProcessing = false,
  disabled = false,
  placeholder = 'Ask anything...',
}: ChatInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [provider, setProvider] = useState<ProviderConfig | null>(() => loadProviderConfig());
  const [dialogOpen, setDialogOpen] = useState(false);
  const [draft, setDraft] = useState<ProviderConfig>(() => defaultProvider());
  const [providerError, setProviderError] = useState<string | null>(null);
  const [savingProvider, setSavingProvider] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    apiFetch(`/api/session/${sessionId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled || !data?.provider) return;
        const current = loadProviderConfig();
        setProvider(current || {
          model: data.provider.model,
          base_url: data.provider.base_url,
          api_key: '',
          context_window: data.provider.context_window,
        });
      })
      .catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [sessionId]);

  useEffect(() => {
    if (!disabled && !isProcessing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled, isProcessing]);

  const handleSend = useCallback(() => {
    if (input.trim() && !disabled) {
      onSend(input);
      setInput('');
    }
  }, [input, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const openProviderDialog = () => {
    setDraft(defaultProvider());
    setProviderError(null);
    setDialogOpen(true);
  };

  const closeProviderDialog = () => {
    if (!savingProvider) setDialogOpen(false);
  };

  const handleSaveProvider = async () => {
    const next = {
      ...draft,
      model: draft.model.trim(),
      base_url: draft.base_url.trim().replace(/\/$/, ''),
      api_key: draft.api_key.trim(),
      context_window: Number(draft.context_window || 200000),
    };
    if (!next.model || !next.base_url || !next.api_key) {
      setProviderError('Model, base URL, and API key are required.');
      return;
    }
    setSavingProvider(true);
    setProviderError(null);
    try {
      saveProviderConfig(next);
      setProvider(next);
      if (sessionId) {
        const res = await apiFetch(`/api/session/${sessionId}/model`, {
          method: 'POST',
          body: JSON.stringify({ provider: next }),
        });
        if (!res.ok) throw new Error(`Provider update failed (${res.status})`);
      }
      setDialogOpen(false);
    } catch (e) {
      setProviderError(e instanceof Error ? e.message : 'Failed to save provider.');
    } finally {
      setSavingProvider(false);
    }
  };

  const providerLabel = provider?.model || 'Configure provider';

  return (
    <Box
      sx={{
        pb: { xs: 2, md: 4 },
        pt: { xs: 1, md: 2 },
        position: 'relative',
        zIndex: 10,
      }}
    >
      <Box sx={{ maxWidth: '880px', mx: 'auto', width: '100%', px: { xs: 0, sm: 1, md: 2 } }}>
        <Box
          className="composer"
          sx={{
            display: 'flex',
            gap: '10px',
            alignItems: 'flex-start',
            bgcolor: 'var(--composer-bg)',
            borderRadius: 'var(--radius-md)',
            p: '12px',
            border: '1px solid var(--border)',
            transition: 'box-shadow 0.2s ease, border-color 0.2s ease',
            '&:focus-within': {
              borderColor: 'var(--accent-yellow)',
              boxShadow: 'var(--focus)',
            },
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={6}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled || isProcessing}
            variant="standard"
            inputRef={inputRef}
            InputProps={{
              disableUnderline: true,
              sx: {
                color: 'var(--text)',
                fontSize: '15px',
                fontFamily: 'inherit',
                padding: 0,
                lineHeight: 1.5,
                minHeight: { xs: '44px', md: '56px' },
                alignItems: 'flex-start',
              },
            }}
            sx={{
              flex: 1,
              '& .MuiInputBase-root': { p: 0, backgroundColor: 'transparent' },
              '& textarea': { resize: 'none', padding: '0 !important' },
            }}
          />
          {isProcessing ? (
            <IconButton
              onClick={onStop}
              sx={{
                mt: 1,
                p: 1.5,
                borderRadius: '10px',
                color: 'var(--muted-text)',
                '&:hover': { bgcolor: 'var(--hover-bg)', color: 'var(--accent-red)' },
              }}
            >
              <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress size={28} thickness={3} sx={{ color: 'inherit', position: 'absolute' }} />
                <StopIcon sx={{ fontSize: 16 }} />
              </Box>
            </IconButton>
          ) : (
            <IconButton
              onClick={handleSend}
              disabled={disabled || !input.trim()}
              sx={{
                mt: 1,
                p: 1,
                borderRadius: '10px',
                color: 'var(--muted-text)',
                '&:hover': { color: 'var(--accent-yellow)', bgcolor: 'var(--hover-bg)' },
                '&.Mui-disabled': { opacity: 0.3 },
              }}
            >
              <ArrowUpwardIcon fontSize="small" />
            </IconButton>
          )}
        </Box>

        <Box
          onClick={openProviderDialog}
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mt: 1.5,
            gap: 0.8,
            opacity: 0.65,
            cursor: 'pointer',
            transition: 'opacity 0.2s',
            '&:hover': { opacity: 1 },
          }}
        >
          <HubOutlinedIcon sx={{ fontSize: 14, color: 'var(--muted-text)' }} />
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
            provider
          </Typography>
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--text)', fontWeight: 600 }}>
            {providerLabel}
          </Typography>
          <SettingsOutlinedIcon sx={{ fontSize: 14, color: 'var(--muted-text)' }} />
        </Box>

        <Dialog open={dialogOpen} onClose={closeProviderDialog} fullWidth maxWidth="sm">
          <DialogTitle>OpenAI-compatible provider</DialogTitle>
          <DialogContent sx={{ display: 'grid', gap: 1.5, pt: '8px !important' }}>
            {providerError && <Alert severity="error">{providerError}</Alert>}
            <TextField
              label="Base URL"
              value={draft.base_url}
              onChange={(e) => setDraft((d) => ({ ...d, base_url: e.target.value }))}
              placeholder="https://api.openai.com/v1"
              size="small"
              required
              fullWidth
            />
            <TextField
              label="Model"
              value={draft.model}
              onChange={(e) => setDraft((d) => ({ ...d, model: e.target.value }))}
              placeholder="gpt-4o-mini"
              size="small"
              required
              fullWidth
            />
            <TextField
              label="API key"
              value={draft.api_key}
              onChange={(e) => setDraft((d) => ({ ...d, api_key: e.target.value }))}
              type="password"
              size="small"
              required
              fullWidth
            />
            <TextField
              label="Context window"
              value={draft.context_window ?? 200000}
              onChange={(e) => setDraft((d) => ({ ...d, context_window: Number(e.target.value) }))}
              type="number"
              size="small"
              fullWidth
            />
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              These settings are saved in this browser and sent to the backend when creating or updating sessions.
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={closeProviderDialog} disabled={savingProvider}>Cancel</Button>
            <Button onClick={handleSaveProvider} disabled={savingProvider} variant="contained">
              {savingProvider ? 'Saving...' : 'Save provider'}
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Box>
  );
}
