use crate::preflight::MtpSupport;
use std::error::Error;
use std::fmt;

/// The decoder only enables speculative execution when it has an executable
/// verifier and a matching proposer. Declaring MTP in config is not enough:
/// MLX exports can omit the MTP tensors entirely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculativeDecodeSupport {
    Unavailable(SpeculativeUnavailableReason),
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

    /// Zero is intentional: callers must take the normal decode path instead
    /// of advertising a speculative depth that cannot be verified.
    pub fn proposal_depth(&self) -> u8 {
        0
    }

    pub fn is_available(&self) -> bool {
        false
    }
}

impl fmt::Display for SpeculativeDecodeSupport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(reason) => write!(formatter, "unavailable: {reason}"),
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
}
