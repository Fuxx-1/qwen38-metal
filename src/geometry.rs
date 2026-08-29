use std::error::Error;
use std::fmt;

pub const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Geometry {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub full_attention_layers: u32,
}

impl Qwen35Geometry {
    pub fn qwen38_27b() -> Self {
        Self {
            hidden_size: 5_120,
            num_hidden_layers: 64,
            num_key_value_heads: 4,
            head_dim: 256,
            full_attention_layers: 16,
        }
    }

    pub fn from_parts(
        hidden_size: u32,
        num_hidden_layers: u32,
        num_key_value_heads: u32,
        head_dim: u32,
        full_attention_layers: u32,
    ) -> Result<Self, GeometryError> {
        for (field, value) in [
            ("hidden_size", hidden_size),
            ("num_hidden_layers", num_hidden_layers),
            ("num_key_value_heads", num_key_value_heads),
            ("head_dim", head_dim),
            ("full_attention_layers", full_attention_layers),
        ] {
            if value == 0 {
                return Err(GeometryError::ZeroDimension(field));
            }
        }

        if full_attention_layers > num_hidden_layers {
            return Err(GeometryError::InvalidFullAttentionLayers {
                full_attention_layers,
                num_hidden_layers,
            });
        }

        Ok(Self {
            hidden_size,
            num_hidden_layers,
            num_key_value_heads,
            head_dim,
            full_attention_layers,
        })
    }

    pub fn kv_elements_per_token(&self) -> Result<u64, GeometryError> {
        checked_mul(
            checked_mul(
                checked_mul(
                    u64::from(self.full_attention_layers),
                    u64::from(self.num_key_value_heads),
                )?,
                u64::from(self.head_dim),
            )?,
            2,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    Bf16,
    Q8,
    Q4,
}

impl KvPrecision {
    pub fn parse(value: &str) -> Result<Self, GeometryError> {
        match value.to_ascii_lowercase().as_str() {
            "bf16" => Ok(Self::Bf16),
            "q8" | "int8" => Ok(Self::Q8),
            "q4" | "int4" => Ok(Self::Q4),
            _ => Err(GeometryError::UnknownPrecision(value.to_owned())),
        }
    }

    pub fn bits_per_element(self) -> u64 {
        match self {
            Self::Bf16 => 16,
            Self::Q8 => 8,
            Self::Q4 => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Q8 => "Q8",
            Self::Q4 => "Q4",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachePlan {
    pub precision: KvPrecision,
    pub context_tokens: u32,
    pub page_tokens: u32,
    pub page_count: u32,
    pub data_bytes: u64,
    pub page_scale_bytes: u64,
    pub total_bytes: u64,
}

impl CachePlan {
    pub fn new(
        geometry: &Qwen35Geometry,
        precision: KvPrecision,
        context_tokens: u32,
        page_tokens: u32,
    ) -> Result<Self, GeometryError> {
        if context_tokens == 0 {
            return Err(GeometryError::ZeroDimension("context_tokens"));
        }
        if page_tokens == 0 {
            return Err(GeometryError::ZeroDimension("page_tokens"));
        }

        let page_count = context_tokens.div_ceil(page_tokens);
        let kv_elements = checked_mul(
            geometry.kv_elements_per_token()?,
            u64::from(context_tokens),
        )?;
        let data_bits = checked_mul(kv_elements, precision.bits_per_element())?;
        let data_bytes = data_bits.div_ceil(8);

        // Each page keeps one FP32 scale per layer, KV head, and K/V pair.
        let page_scale_entries = checked_mul(
            checked_mul(
                checked_mul(
                    u64::from(page_count),
                    u64::from(geometry.full_attention_layers),
                )?,
                u64::from(geometry.num_key_value_heads),
            )?,
            2,
        )?;
        let page_scale_bytes = checked_mul(page_scale_entries, 4)?;

        Ok(Self {
            precision,
            context_tokens,
            page_tokens,
            page_count,
            data_bytes,
            page_scale_bytes,
            total_bytes: checked_add(data_bytes, page_scale_bytes)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M4ProBudget {
    pub unified_memory_bytes: u64,
    pub model_weights_bytes: u64,
    pub runtime_workspace_bytes: u64,
    pub system_reserve_bytes: u64,
}

impl Default for M4ProBudget {
    fn default() -> Self {
        Self {
            unified_memory_bytes: 48 * GIB,
            model_weights_bytes: 17 * GIB,
            runtime_workspace_bytes: 3 * GIB,
            system_reserve_bytes: 12 * GIB,
        }
    }
}

impl M4ProBudget {
    pub fn report(self, cache: &CachePlan) -> Result<MemoryReport, GeometryError> {
        let required_bytes = checked_add(
            checked_add(self.model_weights_bytes, self.runtime_workspace_bytes)?,
            checked_add(self.system_reserve_bytes, cache.total_bytes)?,
        )?;

        Ok(MemoryReport {
            required_bytes,
            available_bytes: self.unified_memory_bytes,
            headroom_bytes: self.unified_memory_bytes.checked_sub(required_bytes),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryReport {
    pub required_bytes: u64,
    pub available_bytes: u64,
    pub headroom_bytes: Option<u64>,
}

impl MemoryReport {
    pub fn fits(self) -> bool {
        self.headroom_bytes.is_some()
    }
}

pub fn format_gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / GIB as f64)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryError {
    ZeroDimension(&'static str),
    InvalidFullAttentionLayers {
        full_attention_layers: u32,
        num_hidden_layers: u32,
    },
    Overflow,
    UnknownPrecision(String),
}

impl fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(field) => write!(formatter, "{field} must be greater than zero"),
            Self::InvalidFullAttentionLayers {
                full_attention_layers,
                num_hidden_layers,
            } => write!(
                formatter,
                "full attention layers ({full_attention_layers}) exceed total layers ({num_hidden_layers})"
            ),
            Self::Overflow => write!(formatter, "cache geometry overflows u64"),
            Self::UnknownPrecision(value) => {
                write!(formatter, "unknown KV precision {value:?}; use bf16, q8, or q4")
            }
        }
    }
}

impl Error for GeometryError {}

fn checked_mul(left: u64, right: u64) -> Result<u64, GeometryError> {
    left.checked_mul(right).ok_or(GeometryError::Overflow)
}

fn checked_add(left: u64, right: u64) -> Result<u64, GeometryError> {
    left.checked_add(right).ok_or(GeometryError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_cache_for_native_262k_is_eight_gib_before_page_scales() {
        let plan = CachePlan::new(
            &Qwen35Geometry::qwen38_27b(),
            KvPrecision::Q8,
            262_144,
            128,
        )
        .unwrap();

        assert_eq!(plan.page_count, 2_048);
        assert_eq!(plan.data_bytes, 8 * GIB);
        assert!(M4ProBudget::default().report(&plan).unwrap().fits());
    }

    #[test]
    fn bf16_cache_exhausts_the_default_48_gib_budget() {
        let plan = CachePlan::new(
            &Qwen35Geometry::qwen38_27b(),
            KvPrecision::Bf16,
            262_144,
            128,
        )
        .unwrap();

        assert_eq!(plan.data_bytes, 16 * GIB);
        assert!(!M4ProBudget::default().report(&plan).unwrap().fits());
    }

    #[test]
    fn q4_cache_uses_four_gib_before_page_scales() {
        let plan = CachePlan::new(
            &Qwen35Geometry::qwen38_27b(),
            KvPrecision::Q4,
            262_144,
            128,
        )
        .unwrap();

        assert_eq!(plan.data_bytes, 4 * GIB);
    }
}
