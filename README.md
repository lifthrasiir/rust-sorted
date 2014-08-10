Rust-sorted
===========

An experimental guaranteed-sorted container.

`SortedVec<T>` and `SortedSlice<'a, T>` are sorted wrappers to `Vec<T>` and `&'a [T]`.
Their methods keep the sortedness or are marked as `unsafe` if they can't.

