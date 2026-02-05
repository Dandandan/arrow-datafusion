## 2025-05-15 - [Optimize bitwise operations in MaybeNullBufferBuilder]
**Learning:** Bit-by-bit loops for shifting nullability information in builders are extremely slow. Using Arrow's bitwise methods like `NullBuffer::slice` and `NullBufferBuilder::append_buffer` can provide massive speedups (up to 660x).
**Action:** Always look for bitwise alternatives to manual loops when dealing with Arrow buffers or builders.
