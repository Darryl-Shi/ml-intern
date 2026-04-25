import { useState, useCallback, useEffect, useRef, KeyboardEvent } from 'react';
import {
  Box,
  TextField,
  IconButton,
  CircularProgress,
  Typography,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Alert,
  Divider,
} from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import StopIcon from '@mui/icons-material/Stop';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import HubOutlinedIcon from '@mui/icons-material/HubOutlined';
import { apiFetch } from '@/utils/api';
import { useUserQuota } from '@/hooks/useUserQuota';
import ClaudeCapDialog from '@/components/ClaudeCapDialog';
import { useAgentStore } from '@/store/agentStore';
import { FIRST_FREE_MODEL_PATH } from '@/utils/model';

// Model configuration
interface ModelOption {
  id: string;
  name: string;
  description: string;
  modelPath: string;
  avatarUrl: string;
  recommended?: boolean;
}

interface CustomProviderInfo {
  model: string;
  base_url: string;
  label?: string | null;
}

const getHfAvatarUrl = (modelId: string) => {
  const org = modelId.split('/')[0];
  return `https://huggingface.co/api/avatars/${org}`;
};

const MODEL_OPTIONS: ModelOption[] = [
  {
    id: 'kimi-k2.6',
    name: 'Kimi K2.6',
    description: 'Novita',
    modelPath: 'moonshotai/Kimi-K2.6',
    avatarUrl: getHfAvatarUrl('moonshotai/Kimi-K2.6'),
    recommended: true,
  },
  {
    id: 'claude-opus',
    name: 'Claude Opus 4.6',
    description: 'Anthropic',
    modelPath: 'anthropic/claude-opus-4-6',
    avatarUrl: 'https://huggingface.co/api/avatars/Anthropic',
    recommended: true,
  },
  {
    id: 'minimax-m2.7',
    name: 'MiniMax M2.7',
    description: 'Novita',
    modelPath: 'MiniMaxAI/MiniMax-M2.7',
    avatarUrl: getHfAvatarUrl('MiniMaxAI/MiniMax-M2.7'),
  },
  {
    id: 'glm-5.1',
    name: 'GLM 5.1',
    description: 'Together',
    modelPath: 'zai-org/GLM-5.1',
    avatarUrl: getHfAvatarUrl('zai-org/GLM-5.1'),
  },
];

const findModelByPath = (path: string): ModelOption | undefined => {
  return MODEL_OPTIONS.find(m => m.modelPath === path);
};

interface ChatInputProps {
  sessionId?: string;
  onSend: (text: string) => void;
  onStop?: () => void;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

const isClaudeModel = (m: ModelOption) => m.modelPath.startsWith('anthropic/');
const firstFreeModel = () => MODEL_OPTIONS.find(m => !isClaudeModel(m)) ?? MODEL_OPTIONS[0];

export default function ChatInput({ sessionId, onSend, onStop, isProcessing = false, disabled = false, placeholder = 'Ask anything...' }: ChatInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [selectedModelId, setSelectedModelId] = useState<string>(MODEL_OPTIONS[0].id);
  const [customProvider, setCustomProvider] = useState<CustomProviderInfo | null>(null);
  const [modelAnchorEl, setModelAnchorEl] = useState<null | HTMLElement>(null);
  const [customDialogOpen, setCustomDialogOpen] = useState(false);
  const [customLabel, setCustomLabel] = useState('');
  const [customModel, setCustomModel] = useState('');
  const [customBaseUrl, setCustomBaseUrl] = useState('');
  const [customApiKey, setCustomApiKey] = useState('');
  const [customError, setCustomError] = useState<string | null>(null);
  const [customSaving, setCustomSaving] = useState(false);
  const { quota, refresh: refreshQuota } = useUserQuota();
  // The daily-cap dialog is triggered from two places: (a) a 429 returned
  // from the chat transport when the user tries to send on Opus over cap —
  // surfaced via the agent-store flag — and (b) nothing else right now
  // (switching models is free). Keeping the open state in the store means
  // the hook layer can flip it without threading props through.
  const claudeQuotaExhausted = useAgentStore((s) => s.claudeQuotaExhausted);
  const setClaudeQuotaExhausted = useAgentStore((s) => s.setClaudeQuotaExhausted);
  const lastSentRef = useRef<string>('');

  // Model is per-session: fetch this tab's current model every time the
  // session changes. Other tabs keep their own selections independently.
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    apiFetch(`/api/session/${sessionId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled) return;
        if (data?.model) {
          if (data.custom_provider) {
            setCustomProvider(data.custom_provider);
            setSelectedModelId('custom');
          } else {
            const model = findModelByPath(data.model);
            if (model) {
              setSelectedModelId(model.id);
              setCustomProvider(null);
            }
          }
        }
      })
      .catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [sessionId]);

  const selectedModel = MODEL_OPTIONS.find(m => m.id === selectedModelId) || MODEL_OPTIONS[0];
  const selectedDisplay = customProvider
    ? {
        name: customProvider.label || customProvider.model,
        description: customProvider.base_url,
        avatarUrl: '',
        isCustom: true,
      }
    : {
        name: selectedModel.name,
        description: selectedModel.description,
        avatarUrl: selectedModel.avatarUrl,
        isCustom: false,
      };

  // Auto-focus the textarea when the session becomes ready
  useEffect(() => {
    if (!disabled && !isProcessing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled, isProcessing]);

  const handleSend = useCallback(() => {
    if (input.trim() && !disabled) {
      lastSentRef.current = input;
      onSend(input);
      setInput('');
    }
  }, [input, disabled, onSend]);

  // When the chat transport reports a Claude-quota 429, restore the typed
  // text so the user doesn't lose their message.
  useEffect(() => {
    if (claudeQuotaExhausted && lastSentRef.current) {
      setInput(lastSentRef.current);
    }
  }, [claudeQuotaExhausted]);

  // Refresh the quota display whenever the session changes (user might
  // have started another tab that spent quota).
  useEffect(() => {
    if (sessionId) refreshQuota();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleModelClick = (event: React.MouseEvent<HTMLElement>) => {
    setModelAnchorEl(event.currentTarget);
  };

  const handleModelClose = () => {
    setModelAnchorEl(null);
  };

  const openCustomDialog = () => {
    handleModelClose();
    setCustomLabel(customProvider?.label || '');
    setCustomModel(customProvider?.model || '');
    setCustomBaseUrl(customProvider?.base_url || '');
    setCustomApiKey('');
    setCustomError(null);
    setCustomDialogOpen(true);
  };

  const closeCustomDialog = () => {
    if (customSaving) return;
    setCustomDialogOpen(false);
    setCustomApiKey('');
    setCustomError(null);
  };

  const handleSelectModel = async (model: ModelOption) => {
    handleModelClose();
    if (!sessionId) return;
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: model.modelPath }),
      });
      if (res.ok) {
        setSelectedModelId(model.id);
        setCustomProvider(null);
      }
    } catch { /* ignore */ }
  };

  const handleSaveCustomProvider = async () => {
    if (!sessionId || customSaving) return;
    const model = customModel.trim();
    const baseUrl = customBaseUrl.trim();
    const apiKey = customApiKey.trim();
    const label = customLabel.trim();
    if (!model || !baseUrl || !apiKey) {
      setCustomError('Model, base URL, and API key are required.');
      return;
    }
    setCustomSaving(true);
    setCustomError(null);
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({
          custom_provider: {
            model,
            base_url: baseUrl,
            api_key: apiKey,
            ...(label ? { label } : {}),
          },
        }),
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => '');
        throw new Error(detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setCustomProvider(data.custom_provider || { model, base_url: baseUrl, label: label || null });
      setSelectedModelId('custom');
      setCustomDialogOpen(false);
      setCustomApiKey('');
    } catch (e) {
      setCustomError(e instanceof Error ? e.message : 'Failed to save custom provider.');
    } finally {
      setCustomSaving(false);
    }
  };

  // Dialog close: just clear the flag. The typed text is already restored.
  const handleCapDialogClose = useCallback(() => {
    setClaudeQuotaExhausted(false);
  }, [setClaudeQuotaExhausted]);

  // "Use a free model" — switch the current session to Kimi (or the first
  // non-Anthropic option) and auto-retry the send that tripped the cap.
  const handleUseFreeModel = useCallback(async () => {
    setClaudeQuotaExhausted(false);
    if (!sessionId) return;
    const free = MODEL_OPTIONS.find(m => m.modelPath === FIRST_FREE_MODEL_PATH)
      ?? firstFreeModel();
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: free.modelPath }),
      });
      if (res.ok) {
        setSelectedModelId(free.id);
        setCustomProvider(null);
        const retryText = lastSentRef.current;
        if (retryText) {
          onSend(retryText);
          setInput('');
          lastSentRef.current = '';
        }
      }
    } catch { /* ignore */ }
  }, [sessionId, onSend, setClaudeQuotaExhausted]);

  // Hide the chip until the user has actually burned quota — an unused
  // Opus session shouldn't populate a counter.
  const claudeChip = (() => {
    if (!quota || quota.claudeUsedToday === 0) return null;
    if (quota.plan === 'free') {
      return quota.claudeRemaining > 0 ? 'Free today' : 'Pro only';
    }
    return `${quota.claudeUsedToday}/${quota.claudeDailyCap} today`;
  })();

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
            }
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
                }
            }}
            sx={{
                flex: 1,
                '& .MuiInputBase-root': {
                    p: 0,
                    backgroundColor: 'transparent',
                },
                '& textarea': {
                    resize: 'none',
                    padding: '0 !important',
                }
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
                transition: 'all 0.2s',
                position: 'relative',
                '&:hover': {
                  bgcolor: 'var(--hover-bg)',
                  color: 'var(--accent-red)',
                },
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
                transition: 'all 0.2s',
                '&:hover': {
                  color: 'var(--accent-yellow)',
                  bgcolor: 'var(--hover-bg)',
                },
                '&.Mui-disabled': {
                  opacity: 0.3,
                },
              }}
            >
              <ArrowUpwardIcon fontSize="small" />
            </IconButton>
          )}
        </Box>

        {/* Powered By Badge */}
        <Box
          onClick={handleModelClick}
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mt: 1.5,
            gap: 0.8,
            opacity: 0.6,
            cursor: 'pointer',
            transition: 'opacity 0.2s',
            '&:hover': {
              opacity: 1
            }
          }}
        >
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
            powered by
          </Typography>
          {selectedDisplay.isCustom ? (
            <HubOutlinedIcon sx={{ fontSize: 14, color: 'var(--muted-text)' }} />
          ) : (
            <img
              src={selectedDisplay.avatarUrl}
              alt={selectedDisplay.name}
              style={{ height: '14px', width: '14px', objectFit: 'contain', borderRadius: '2px' }}
            />
          )}
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--text)', fontWeight: 600, letterSpacing: '0.02em' }}>
            {selectedDisplay.name}
          </Typography>
          <ArrowDropDownIcon sx={{ fontSize: '14px', color: 'var(--muted-text)' }} />
        </Box>

        {/* Model Selection Menu */}
        <Menu
          anchorEl={modelAnchorEl}
          open={Boolean(modelAnchorEl)}
          onClose={handleModelClose}
          anchorOrigin={{
            vertical: 'top',
            horizontal: 'center',
          }}
          transformOrigin={{
            vertical: 'bottom',
            horizontal: 'center',
          }}
          slotProps={{
            paper: {
              sx: {
                bgcolor: 'var(--panel)',
                border: '1px solid var(--divider)',
                mb: 1,
                maxHeight: '400px',
              }
            }
          }}
        >
          {MODEL_OPTIONS.map((model) => (
            <MenuItem
              key={model.id}
              onClick={() => handleSelectModel(model)}
              selected={selectedModelId === model.id}
              sx={{
                py: 1.5,
                '&.Mui-selected': {
                  bgcolor: 'rgba(255,255,255,0.05)',
                }
              }}
            >
              <ListItemIcon>
                <img
                  src={model.avatarUrl}
                  alt={model.name}
                  style={{ width: 24, height: 24, borderRadius: '4px', objectFit: 'cover' }}
                />
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {model.name}
                    {model.recommended && (
                      <Chip
                        label="Recommended"
                        size="small"
                        sx={{
                          height: '18px',
                          fontSize: '10px',
                          bgcolor: 'var(--accent-yellow)',
                          color: '#000',
                          fontWeight: 600,
                        }}
                      />
                    )}
                    {isClaudeModel(model) && claudeChip && (
                      <Chip
                        label={claudeChip}
                        size="small"
                        sx={{
                          height: '18px',
                          fontSize: '10px',
                          bgcolor: 'rgba(255,255,255,0.08)',
                          color: 'var(--muted-text)',
                          fontWeight: 600,
                        }}
                      />
                    )}
                  </Box>
                }
                secondary={model.description}
                secondaryTypographyProps={{
                  sx: { fontSize: '12px', color: 'var(--muted-text)' }
                }}
              />
            </MenuItem>
          ))}
          <Divider sx={{ borderColor: 'var(--divider)' }} />
          <MenuItem
            onClick={openCustomDialog}
            selected={selectedModelId === 'custom'}
            sx={{
              py: 1.5,
              '&.Mui-selected': {
                bgcolor: 'rgba(255,255,255,0.05)',
              }
            }}
          >
            <ListItemIcon>
              <AddCircleOutlineIcon sx={{ color: 'var(--muted-text)' }} />
            </ListItemIcon>
            <ListItemText
              primary={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  Custom provider
                  {customProvider && (
                    <Chip
                      label="Active"
                      size="small"
                      sx={{
                        height: '18px',
                        fontSize: '10px',
                        bgcolor: 'rgba(255,255,255,0.08)',
                        color: 'var(--muted-text)',
                        fontWeight: 600,
                      }}
                    />
                  )}
                </Box>
              }
              secondary={customProvider?.model || 'OpenAI-compatible endpoint'}
              secondaryTypographyProps={{
                sx: { fontSize: '12px', color: 'var(--muted-text)' }
              }}
            />
          </MenuItem>
        </Menu>

        <Dialog
          open={customDialogOpen}
          onClose={closeCustomDialog}
          fullWidth
          maxWidth="sm"
          PaperProps={{
            sx: {
              bgcolor: 'var(--panel)',
              border: '1px solid var(--divider)',
            },
          }}
        >
          <DialogTitle sx={{ fontSize: '1rem', fontWeight: 700 }}>
            Custom provider
          </DialogTitle>
          <DialogContent sx={{ display: 'grid', gap: 1.5, pt: '8px !important' }}>
            {customError && (
              <Alert severity="error" sx={{ fontSize: '0.78rem' }}>
                {customError}
              </Alert>
            )}
            <TextField
              label="Label"
              value={customLabel}
              onChange={(e) => setCustomLabel(e.target.value)}
              placeholder="Local vLLM"
              size="small"
              fullWidth
            />
            <TextField
              label="Model"
              value={customModel}
              onChange={(e) => setCustomModel(e.target.value)}
              placeholder="meta-llama/Llama-3.1-8B-Instruct"
              size="small"
              required
              fullWidth
            />
            <TextField
              label="Base URL"
              value={customBaseUrl}
              onChange={(e) => setCustomBaseUrl(e.target.value)}
              placeholder="https://api.example.com/v1"
              size="small"
              required
              fullWidth
            />
            <TextField
              label="API key"
              value={customApiKey}
              onChange={(e) => setCustomApiKey(e.target.value)}
              type="password"
              size="small"
              required
              fullWidth
            />
            <Typography variant="caption" sx={{ color: 'var(--muted-text)' }}>
              The key is kept only for this backend session and is never saved in your browser.
            </Typography>
          </DialogContent>
          <DialogActions sx={{ px: 3, pb: 2 }}>
            <Button onClick={closeCustomDialog} disabled={customSaving} sx={{ textTransform: 'none' }}>
              Cancel
            </Button>
            <Button
              onClick={handleSaveCustomProvider}
              disabled={customSaving}
              variant="contained"
              sx={{ textTransform: 'none' }}
            >
              {customSaving ? 'Saving...' : 'Use provider'}
            </Button>
          </DialogActions>
        </Dialog>

        <ClaudeCapDialog
          open={claudeQuotaExhausted}
          plan={quota?.plan ?? 'free'}
          cap={quota?.claudeDailyCap ?? 1}
          onClose={handleCapDialogClose}
          onUseFreeModel={handleUseFreeModel}
        />
      </Box>
    </Box>
  );
}
