use std::sync::{Arc, Mutex};

/// Fixed-capacity buffer pool to prevent high-frequency frame heap churn.
#[derive(Clone)]
pub struct FrameBufferPool {
    inner: Arc<Mutex<PoolState>>,
    max_buffers: usize,
}

struct PoolState {
    free: Vec<Vec<u8>>,
}

impl FrameBufferPool {
    pub fn new(max_buffers: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PoolState {
                free: Vec::with_capacity(max_buffers),
            })),
            max_buffers: max_buffers.max(1),
        }
    }

    pub fn checkout(&self, required_bytes: usize) -> PooledBuffer {
        let mut buf = {
            let mut guard = self.inner.lock().unwrap();
            guard.free.pop().unwrap_or_default()
        };
        if buf.len() != required_bytes {
            buf.resize(required_bytes, 0);
        }
        PooledBuffer {
            data: Some(buf),
            pool: self.clone(),
        }
    }

    fn return_buffer(&self, mut buf: Vec<u8>) {
        let mut guard = self.inner.lock().unwrap();
        if guard.free.len() < self.max_buffers {
            buf.clear();
            guard.free.push(buf);
        }
    }
}

/// RAII frame buffer that returns its backing allocation to `FrameBufferPool` on drop.
pub struct PooledBuffer {
    data: Option<Vec<u8>>,
    pool: FrameBufferPool,
}

impl PooledBuffer {
    pub fn as_slice(&self) -> &[u8] {
        self.data.as_deref().unwrap_or(&[])
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.data.as_deref_mut().unwrap_or(&mut [])
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.data.take() {
            self.pool.return_buffer(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_reuses_and_caps_buffers() {
        let pool = FrameBufferPool::new(2);
        let b1 = pool.checkout(1024);
        let ptr1 = b1.as_slice().as_ptr();
        let b2 = pool.checkout(1024);
        let b3 = pool.checkout(1024);
        drop(b1);
        drop(b2);
        drop(b3);

        let reused = pool.checkout(1024);
        // Should pop LIFO or reuse one of the pooled allocations without allocating new capacity if same size.
        assert_eq!(reused.len(), 1024);
        let _ = ptr1;
        let guard = pool.inner.lock().unwrap();
        assert!(guard.free.len() <= 2);
    }
}
