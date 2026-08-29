use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageSpan {
    pub start_token: u32,
    pub end_token_exclusive: u32,
    pub first_page: u32,
    pub last_page: u32,
}

#[derive(Debug, Clone)]
pub struct LogicalPageTable {
    max_tokens: u32,
    page_tokens: u32,
    sequence_tokens: u32,
    pages: Vec<Option<u32>>,
    next_page_id: u32,
}

impl LogicalPageTable {
    pub fn new(max_tokens: u32, page_tokens: u32) -> Result<Self, PageTableError> {
        if max_tokens == 0 || page_tokens == 0 {
            return Err(PageTableError::ZeroCapacity);
        }

        let page_count = max_tokens.div_ceil(page_tokens);
        Ok(Self {
            max_tokens,
            page_tokens,
            sequence_tokens: 0,
            pages: vec![None; page_count as usize],
            next_page_id: 0,
        })
    }

    pub fn append(&mut self, token_count: u32) -> Result<PageSpan, PageTableError> {
        if token_count == 0 {
            return Err(PageTableError::EmptyAppend);
        }

        let start_token = self.sequence_tokens;
        let end_token_exclusive =
            start_token
                .checked_add(token_count)
                .ok_or(PageTableError::CapacityExceeded {
                    requested_end: u32::MAX,
                    max_tokens: self.max_tokens,
                })?;
        if end_token_exclusive > self.max_tokens {
            return Err(PageTableError::CapacityExceeded {
                requested_end: end_token_exclusive,
                max_tokens: self.max_tokens,
            });
        }

        let first_page = start_token / self.page_tokens;
        let last_page = (end_token_exclusive - 1) / self.page_tokens;
        for logical_page in first_page..=last_page {
            self.allocate(logical_page)?;
        }
        self.sequence_tokens = end_token_exclusive;

        Ok(PageSpan {
            start_token,
            end_token_exclusive,
            first_page,
            last_page,
        })
    }

    pub fn allocated_pages(&self) -> usize {
        self.pages.iter().flatten().count()
    }

    pub fn page_for_token(&self, token: u32) -> Option<u32> {
        if token >= self.sequence_tokens {
            return None;
        }

        self.pages
            .get((token / self.page_tokens) as usize)
            .and_then(|page| *page)
    }

    fn allocate(&mut self, logical_page: u32) -> Result<(), PageTableError> {
        let slot =
            self.pages
                .get_mut(logical_page as usize)
                .ok_or(PageTableError::CapacityExceeded {
                    requested_end: logical_page.saturating_mul(self.page_tokens),
                    max_tokens: self.max_tokens,
                })?;
        if slot.is_none() {
            *slot = Some(self.next_page_id);
            self.next_page_id = self.next_page_id.saturating_add(1);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageTableError {
    ZeroCapacity,
    EmptyAppend,
    CapacityExceeded { requested_end: u32, max_tokens: u32 },
}

impl fmt::Display for PageTableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => {
                write!(
                    formatter,
                    "page table capacity and page size must be non-zero"
                )
            }
            Self::EmptyAppend => write!(formatter, "cannot append an empty token range"),
            Self::CapacityExceeded {
                requested_end,
                max_tokens,
            } => write!(
                formatter,
                "token range ending at {requested_end} exceeds page table capacity {max_tokens}"
            ),
        }
    }
}

impl Error for PageTableError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_are_materialized_only_when_tokens_arrive() {
        let mut table = LogicalPageTable::new(262_144, 128).unwrap();
        assert_eq!(table.allocated_pages(), 0);

        table.append(127).unwrap();
        assert_eq!(table.allocated_pages(), 1);
        assert_eq!(table.page_for_token(0), Some(0));

        table.append(1).unwrap();
        assert_eq!(table.allocated_pages(), 1);

        table.append(1).unwrap();
        assert_eq!(table.allocated_pages(), 2);
        assert_eq!(table.page_for_token(128), Some(1));
    }

    #[test]
    fn append_rejects_ranges_past_the_context_limit() {
        let mut table = LogicalPageTable::new(128, 128).unwrap();
        table.append(128).unwrap();

        assert!(matches!(
            table.append(1),
            Err(PageTableError::CapacityExceeded { .. })
        ));
    }
}
