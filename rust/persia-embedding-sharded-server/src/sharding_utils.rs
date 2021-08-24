#[inline]
fn ncb(shard_amount: usize) -> usize {
    shard_amount.trailing_zeros() as usize
}

#[inline]
pub fn ptr_size_bits() -> usize {
    std::mem::size_of::<usize>() * 8
}

#[inline]
pub fn num_shards_to_shift(num_shards: usize) -> usize {
    ptr_size_bits() - ncb(num_shards)
}

#[inline]
pub fn id_to_shard(hashed_id: u64, num_shards_shift: usize) -> u64 {
    (hashed_id << 7) >> num_shards_shift
}

#[cfg(test)]
mod test {
    use super::{id_to_shard, num_shards_to_shift};

    #[test]
    fn test_id_to_shard() {
        let id = 59102438219048053_u64;
        let hashed_id = id.wrapping_mul(11400714819323198549u64);
        let shards_shift = num_shards_to_shift(8 * 32);
        assert_eq!(shards_shift, 56);
        let shard_idx = id_to_shard(hashed_id, shards_shift);
        assert_eq!(shard_idx, 242);
    }
}
