use crate::preflight::MtpSupport;
use std::error::Error;
use std::fmt;

/// The decoder only enables speculative execution when it has an executable
/// verifier and a matching proposer. Declaring MTP in config is not enough:
/// MLX exports can omit the MTP tensors entirely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculativeDecodeSupport {
    Unavailable(SpeculativeUnavailableReason),
    Available {
        configured_layers: u32,
        tensor_count: usize,
        draft_tokens: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculativeUnavailableReason {
    NotDeclared,
    WeightsMissing {
        configured_layers: u32,
    },
    VerifierNotImplemented {
        configured_layers: u32,
        tensor_count: usize,
    },
    AdapterNotConfigured {
        configured_layers: u32,
        tensor_count: usize,
    },
}

impl SpeculativeDecodeSupport {
    pub fn from_mtp_support(support: &MtpSupport) -> Self {
        Self::Unavailable(match support {
            MtpSupport::NotDeclared => SpeculativeUnavailableReason::NotDeclared,
            MtpSupport::DeclaredButWeightsMissing { configured_layers } => {
                SpeculativeUnavailableReason::WeightsMissing {
                    configured_layers: *configured_layers,
                }
            }
            MtpSupport::Available {
                configured_layers,
                tensor_count,
            } => SpeculativeUnavailableReason::VerifierNotImplemented {
                configured_layers: *configured_layers,
                tensor_count: *tensor_count,
            },
        })
    }

    /// Builds the advertised capability after both sides of a standalone
    /// Qwen MTP pair have been loaded. The target export commonly declares
    /// MTP but intentionally omits the adapter tensors, so target support by
    /// itself is not sufficient to enable speculation.
    pub fn from_loaded_adapter(target: &MtpSupport, adapter: &MtpSupport, block_size: u8) -> Self {
        let target_layers = match target {
            MtpSupport::NotDeclared => 0,
            MtpSupport::DeclaredButWeightsMissing { configured_layers }
            | MtpSupport::Available {
                configured_layers, ..
            } => *configured_layers,
        };
        let adapter_capability = match adapter {
            MtpSupport::Available {
                configured_layers,
                tensor_count,
            } => Some((*configured_layers, *tensor_count)),
            _ => None,
        };
        if target_layers > 0 && block_size >= 2 {
            if let Some((configured_layers, tensor_count)) = adapter_capability {
                return Self::Available {
                    configured_layers: configured_layers.min(target_layers),
                    tensor_count,
                    draft_tokens: block_size.saturating_sub(1),
                };
            }
        }

        let (configured_layers, tensor_count) = match target {
            MtpSupport::Available {
                configured_layers,
                tensor_count,
            } => (*configured_layers, *tensor_count),
            MtpSupport::DeclaredButWeightsMissing { configured_layers } => (*configured_layers, 0),
            MtpSupport::NotDeclared => (0, 0),
        };
        Self::Unavailable(SpeculativeUnavailableReason::AdapterNotConfigured {
            configured_layers,
            tensor_count,
        })
    }

    /// Zero is intentional: callers must take the normal decode path instead
    /// of advertising a speculative depth that cannot be verified.
    pub fn proposal_depth(&self) -> u8 {
        match self {
            Self::Available { draft_tokens, .. } => *draft_tokens,
            Self::Unavailable(_) => 0,
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }
}

impl fmt::Display for SpeculativeDecodeSupport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(reason) => write!(formatter, "unavailable: {reason}"),
            Self::Available {
                configured_layers,
                tensor_count,
                draft_tokens,
            } => write!(
                formatter,
                "available: {configured_layers} layer(s), {tensor_count} tensors, {draft_tokens} draft token(s)"
            ),
        }
    }
}

impl fmt::Display for SpeculativeUnavailableReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotDeclared => write!(formatter, "the model does not declare MTP"),
            Self::WeightsMissing { configured_layers } => write!(
                formatter,
                "the model declares {configured_layers} MTP layer(s), but their tensors are absent"
            ),
            Self::VerifierNotImplemented {
                configured_layers,
                tensor_count,
            } => write!(
                formatter,
                "the model exposes {tensor_count} tensors for {configured_layers} MTP layer(s), but no verifier has been loaded"
            ),
            Self::AdapterNotConfigured {
                configured_layers,
                tensor_count,
            } => write!(
                formatter,
                "the target declares {configured_layers} MTP layer(s) ({tensor_count} inline tensors), but no matching standalone adapter is configured"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MtpController {
    min_depth: u8,
    max_depth: u8,
    current_depth: u8,
    acceptance_ema: Option<f32>,
}

impl MtpController {
    pub fn new(min_depth: u8, max_depth: u8, initial_depth: u8) -> Result<Self, MtpError> {
        if min_depth == 0 || min_depth > max_depth {
            return Err(MtpError::InvalidDepthRange {
                min_depth,
                max_depth,
            });
        }
        if !(min_depth..=max_depth).contains(&initial_depth) {
            return Err(MtpError::InitialDepthOutOfRange {
                initial_depth,
                min_depth,
                max_depth,
            });
        }

        Ok(Self {
            min_depth,
            max_depth,
            current_depth: initial_depth,
            acceptance_ema: None,
        })
    }

    pub fn recommended_depth(&self) -> u8 {
        self.current_depth
    }

    pub fn acceptance_ema(&self) -> Option<f32> {
        self.acceptance_ema
    }

    pub fn observe(&mut self, proposed_tokens: u8, accepted_tokens: u8) -> Result<u8, MtpError> {
        if proposed_tokens == 0 || accepted_tokens > proposed_tokens {
            return Err(MtpError::InvalidAcceptance {
                proposed_tokens,
                accepted_tokens,
            });
        }

        let acceptance = f32::from(accepted_tokens) / f32::from(proposed_tokens);
        let ema = match self.acceptance_ema {
            Some(previous) => previous * 0.75 + acceptance * 0.25,
            None => acceptance,
        };
        self.acceptance_ema = Some(ema);

        if ema >= 0.80 && self.current_depth < self.max_depth {
            self.current_depth += 1;
        } else if ema <= 0.45 && self.current_depth > self.min_depth {
            self.current_depth -= 1;
        }

        Ok(self.current_depth)
    }
}

/// Returns the longest greedy prefix of draft tokens that agrees with the
/// target verifier. The target row at that same index is the next bonus token.
pub(crate) fn accepted_token_count(
    draft_tokens: &[u32],
    target_tokens: &[u32],
    draft_count: usize,
) -> usize {
    draft_tokens
        .iter()
        .zip(target_tokens.iter())
        .take(draft_count)
        .take_while(|(draft, target)| draft == target)
        .count()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpError {
    InvalidDepthRange {
        min_depth: u8,
        max_depth: u8,
    },
    InitialDepthOutOfRange {
        initial_depth: u8,
        min_depth: u8,
        max_depth: u8,
    },
    InvalidAcceptance {
        proposed_tokens: u8,
        accepted_tokens: u8,
    },
}

impl fmt::Display for MtpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDepthRange {
                min_depth,
                max_depth,
            } => write!(
                formatter,
                "invalid MTP depth range {min_depth}..={max_depth}"
            ),
            Self::InitialDepthOutOfRange {
                initial_depth,
                min_depth,
                max_depth,
            } => write!(
                formatter,
                "initial MTP depth {initial_depth} is outside {min_depth}..={max_depth}"
            ),
            Self::InvalidAcceptance {
                proposed_tokens,
                accepted_tokens,
            } => write!(
                formatter,
                "accepted MTP tokens ({accepted_tokens}) exceed proposed tokens ({proposed_tokens})"
            ),
        }
    }
}

impl Error for MtpError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_acceptance_increases_draft_depth() {
        let mut controller = MtpController::new(1, 3, 2).unwrap();
        assert_eq!(controller.observe(2, 2).unwrap(), 3);
    }

    #[test]
    fn repeated_low_acceptance_reduces_draft_depth() {
        let mut controller = MtpController::new(1, 3, 3).unwrap();
        assert_eq!(controller.observe(3, 0).unwrap(), 2);
        assert_eq!(controller.observe(2, 0).unwrap(), 1);
    }

    #[test]
    fn missing_mtp_weights_never_advertise_speculation() {
        let support =
            SpeculativeDecodeSupport::from_mtp_support(&MtpSupport::DeclaredButWeightsMissing {
                configured_layers: 1,
            });
        assert!(!support.is_available());
        assert_eq!(support.proposal_depth(), 0);
        assert!(support.to_string().contains("tensors are absent"));
    }

    #[test]
    fn loaded_adapter_advertises_matching_draft_depth() {
        let support = SpeculativeDecodeSupport::from_loaded_adapter(
            &MtpSupport::DeclaredButWeightsMissing {
                configured_layers: 1,
            },
            &MtpSupport::Available {
                configured_layers: 1,
                tensor_count: 31,
            },
            3,
        );
        assert_eq!(support.proposal_depth(), 2);
        assert!(support.is_available());
        assert!(support.to_string().contains("31 tensors"));
    }

    #[test]
    fn adapter_requires_a_two_token_block() {
        let support = SpeculativeDecodeSupport::from_loaded_adapter(
            &MtpSupport::DeclaredButWeightsMissing {
                configured_layers: 1,
            },
            &MtpSupport::Available {
                configured_layers: 1,
                tensor_count: 31,
            },
            1,
        );
        assert!(!support.is_available());
        assert_eq!(support.proposal_depth(), 0);
    }

    #[test]
    fn accepted_tokens_stop_at_first_mismatch() {
        assert_eq!(accepted_token_count(&[4, 5, 6], &[4, 9, 6, 7], 3), 1);
        assert_eq!(accepted_token_count(&[4, 5], &[4, 5, 6], 2), 2);
        assert_eq!(accepted_token_count(&[4, 5], &[4], 2), 1);
    }
}
