use std::error::Error;
use std::fmt;

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
}
