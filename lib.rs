// Rust-sorted: Guaranteed-sorted containers.
// Copyright (c) 2014, Kang Seonghoon.
// See README.md and LICENSE.txt for details.

#![feature(phase, default_type_params)]

extern crate core;
#[cfg(test)] extern crate quickcheck;
#[cfg(test)] #[phase(plugin)] extern crate quickcheck_macros;

use std::{slice, vec, fmt, hash};
use std::mem::{forget, swap};
use std::default::Default;
#[cfg(test)] use quickcheck::TestResult;

fn is_sorted<T: Ord>(v: &[T]) -> bool {
    if v.len() < 2 {
        true
    } else {
        v.windows(2).all(|win| win[0] <= win[1])
    }
}

fn append_garbages<T>(a: &mut Vec<T>, n: uint) {
    let len = a.len();
    a.reserve_additional(n);
    unsafe { a.set_len(len + n); }
}

fn replace_garbage<T>(a: &mut T, mut v: T) {
    swap(a, &mut v);
    unsafe { forget(v); }
}

/// Returns the first index `i` such that `v[i]` is no `Less` than the target,
/// or `v.len()` if there is no such `i`.
/// Similar to `vec::bsearch` but `v[i]` (if any) needs not be `Equal` to the target.
fn bsearch_no_less<T>(v: &[T], f: |&T| -> Ordering) -> uint {
    let mut base = 0;
    let mut limit = v.len();
    while limit != 0 { // invariant: v[base-1] (if any) < target <= v[base+limit] (if any)
        let ix = base + (limit >> 1);
        if f(&v[ix]) == Less {
            base = ix + 1;
            limit -= 1;
        }
        limit >>= 1;
    }
    base
}

#[quickcheck]
fn test_bsearch_no_less(mut v: Vec<uint>, x: uint) -> bool {
    v.sort();
    let i = bsearch_no_less(v.as_slice(), |&y| y.cmp(&x));
    v.slice_to(i).iter().all(|&y| y < x) && v.slice_from(i).iter().all(|&y| y >= x)
}

/// Returns the first index `i` such that `v[i]` is `Greater` than the target,
/// or `v.len()` if there is no such `i`.
/// Similar to `vec::bsearch` but `v[i]` (if any) needs not be `Equal` to the target.
fn bsearch_greater<T>(v: &[T], f: |&T| -> Ordering) -> uint {
    let mut base = 0;
    let mut limit = v.len();
    while limit != 0 { // invariant: v[base-1] (if any) <= target < v[base+limit] (if any)
        let ix = base + (limit >> 1);
        if f(&v[ix]) != Greater {
            base = ix + 1;
            limit -= 1;
        }
        limit >>= 1;
    }
    base
}

#[quickcheck]
fn test_bsearch_greater(mut v: Vec<uint>, x: uint) -> bool {
    v.sort();
    let i = bsearch_greater(v.as_slice(), |&y| y.cmp(&x));
    v.slice_to(i).iter().all(|&y| y <= x) && v.slice_from(i).iter().all(|&y| y > x)
}

fn merge_to_sorted_vec<T: Ord + Clone>(a: &mut Vec<T>, b: &[T]) {
    let alen = a.len();
    let blen = b.len();
    append_garbages(a, blen);

    //  0                       i           i+j
    // +-----------------------+-----------+------+
    // |        sorted         |  garbage  |sorted|
    // +-----------------------+-----------+------+
    let mut i = alen;
    let mut j = blen;
    while i > 0 && j > 0 {
        if a.as_slice()[i-1].cmp(&b[j-1]) == Greater { // a goes first
            a.as_mut_slice().swap(i-1, i+j-1);
            i -= 1;
        } else { // b goes first
            replace_garbage(&mut a.as_mut_slice()[i+j-1], b[j-1].clone());
            j -= 1;
        }
    }
    while j > 0 { // all elements in b are smaller than a[0]
        replace_garbage(&mut a.as_mut_slice()[i+j-1], b[j-1].clone());
        j -= 1;
    }
}

#[quickcheck]
fn test_merge_to_sorted_vec(mut a: Vec<uint>, mut b: Vec<uint>) -> bool {
    a.sort();
    b.sort();
    let mut ab = a + b;
    ab.sort();

    merge_to_sorted_vec(&mut a, b.as_slice());
    a == ab
}

fn merge_into_sorted_vec<T: Ord>(a: &mut Vec<T>, mut b: Vec<T>) {
    let alen = a.len();
    let blen = b.len();
    append_garbages(a, blen);

    //  0                       i           i+b.len()
    // +-----------------------+-----------+------+
    // |        sorted         |  garbage  |sorted|
    // +-----------------------+-----------+------+
    let mut i = alen;
    while i > 0 && !b.is_empty() {
        if a.as_slice()[i-1].cmp(b.last().unwrap()) == Greater { // a goes first
            a.as_mut_slice().swap(i-1, i+b.len()-1);
            i -= 1;
        } else { // b goes first
            let v = b.pop().unwrap();
            replace_garbage(&mut a.as_mut_slice()[i+b.len()], v);
        }
    }
    while !b.is_empty() { // all elements in b are smaller than a[0]
        let v = b.pop().unwrap();
        replace_garbage(&mut a.as_mut_slice()[i+b.len()], v);
    }
}

#[quickcheck]
fn test_merge_into_sorted_vec(mut a: Vec<uint>, mut b: Vec<uint>) -> bool {
    a.sort();
    b.sort();
    let mut ab = a + b;
    ab.sort();

    merge_into_sorted_vec(&mut a, b);
    a == ab
}

fn insert_many<T: Clone>(a: &mut Vec<T>, k: uint, n: uint, v: &T) {
    let len = a.len();
    append_garbages(a, n);
    for i in range(k, len).rev() {
        a.as_mut_slice().swap(i, i + n);
    }
    for i in range(k, k + n) {
        replace_garbage(&mut a.as_mut_slice()[i], v.clone());
    }
}

#[quickcheck]
fn test_insert_many(mut a: Vec<uint>, (k, n): (uint, uint), v: uint) -> TestResult {
    if k > a.len() { return TestResult::discard(); }

    let mut b = a.as_slice().slice_to(k).to_vec();
    b.grow(n, &v);
    b.push_all(a.as_slice().slice_from(k));

    insert_many(&mut a, k, n, &v);
    TestResult::from_bool(a == b)
}

fn is_disjoint<T: Ord>(a: &[T], b: &[T]) -> bool {
    let alen = a.len();
    let blen = b.len();
    let mut i = 0;
    let mut j = 0;
    while i < alen && j < blen {
        match a[i].cmp(&b[j]) {
            Less => { i += 1; } // a[i] is not in b
            Equal => { return false; }
            Greater => { j += 1; } // b[j] is not in a
        }
    }
    true
}

#[quickcheck]
fn test_is_disjoint_random(mut a: Vec<uint>, mut b: Vec<uint>) -> bool {
    use std::collections::HashSet;

    a.sort();
    b.sort();
    let aset: HashSet<uint> = a.iter().map(|&v| v).collect();
    let bset: HashSet<uint> = b.iter().map(|&v| v).collect();
    is_disjoint(a.as_slice(), b.as_slice()) == aset.is_disjoint(&bset)
}

#[quickcheck]
fn test_is_disjoint_always(a: Vec<uint>, b: Vec<uint>) -> bool {
    let mut a: Vec<uint> = a.move_iter().map(|v| 2*v  ).collect();
    let mut b: Vec<uint> = b.move_iter().map(|v| 2*v+1).collect();
    a.sort();
    b.sort();
    is_disjoint(a.as_slice(), b.as_slice())
}

#[quickcheck]
fn test_is_disjoint_never(a: Vec<uint>, b: Vec<uint>, c: Vec<uint>) -> TestResult {
    if b.is_empty() { return TestResult::discard(); }

    let mut ab = a + b;
    let mut bc = b + c;
    ab.sort();
    bc.sort();
    TestResult::from_bool(!is_disjoint(ab.as_slice(), bc.as_slice()))
}

fn is_subset<T: Ord>(a: &[T], b: &[T]) -> bool {
    let alen = a.len();
    let blen = b.len();
    let mut i = 0;
    let mut j = 0;
    while i < alen && j < blen {
        match a[i].cmp(&b[j]) {
            Less => { return false; } // a[i] is not in b
            Equal => { i += 1; j += 1; }
            Greater => { j += 1; } // b[j] is not in a
        }
    }
    if i < alen { // a[i]..a[alen-1] is not in b
        false
    } else { // b[j]..b[blen-1] is not in a, if not empty
        true
    }
}

#[quickcheck]
fn test_is_subset_random(mut a: Vec<uint>, mut b: Vec<uint>) -> bool {
    use std::collections::HashSet;

    a.sort();
    b.sort();
    let aset: HashSet<uint> = a.iter().map(|&v| v).collect();
    let bset: HashSet<uint> = b.iter().map(|&v| v).collect();
    is_subset(a.as_slice(), b.as_slice()) == aset.is_subset(&bset)
}

#[quickcheck]
fn test_is_subset_always(mut a: Vec<uint>, b: Vec<uint>) -> bool {
    let mut ab = a + b;
    a.sort();
    ab.sort();
    is_subset(a.as_slice(), ab.as_slice())
}

#[quickcheck]
fn test_is_subset_never(a: Vec<uint>, b: Vec<uint>, c: Vec<uint>) -> TestResult {
    if b.is_empty() { return TestResult::discard(); }

    let a: Vec<uint> = a.move_iter().map(|v| 3*v  ).collect();
    let b: Vec<uint> = b.move_iter().map(|v| 3*v+1).collect();
    let c: Vec<uint> = c.move_iter().map(|v| 3*v+2).collect();
    let mut ab = a + b;
    let mut ac = a + c;
    ab.sort();
    ac.sort();
    TestResult::from_bool(!is_subset(ab.as_slice(), ac.as_slice()))
}

/// A sorted slice. Equivalent to `&'a [T]` except that it is known to be sorted.
pub struct SortedSlice<'a, T> { inner: &'a [T] }

impl<'a, T: Ord> SortedSlice<'a, T> {
    pub fn from_slice(values: &'a [T]) -> Result<SortedSlice<'a, T>, &'a [T]> {
        if is_sorted(values) {
            Ok(SortedSlice { inner: values })
        } else {
            Err(values)
        }
    }

    pub fn from_ref(s: &'a T) -> SortedSlice<'a, T> {
        SortedSlice { inner: slice::ref_slice(s) }
    }

    pub fn unwrap(self) -> &'a [T] {
        self.inner
    }
}

impl<'a, T: Ord> /*ImmutableVector<'a, T> for*/ SortedSlice<'a, T> {
    pub fn slice(&self, start: uint, end: uint) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.slice(start, end) }
    }

    pub fn slice_from(&self, start: uint) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.slice_from(start) }
    }

    pub fn slice_to(&self, end: uint) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.slice_to(end) }
    }

    pub fn split_at(&self, mid: uint) -> (SortedSlice<'a, T>, SortedSlice<'a, T>) {
        let (left, right) = self.inner.split_at(mid);
        (SortedSlice { inner: left }, SortedSlice { inner: right })
    }

    pub fn iter(&self) -> slice::Items<'a, T> {
        self.inner.iter()
    }

    pub fn split(self, pred: |&T|: 'a -> bool) -> slice::Splits<'a, T> {
        self.inner.split(pred)
    }

    pub fn splitn(self, n: uint, pred: |&T|: 'a -> bool) -> core::slice::SplitsN<'a, T> {
        self.inner.splitn(n, pred)
    }

    pub fn rsplitn(self, n: uint, pred: |&T|: 'a -> bool) -> core::slice::SplitsN<'a, T> {
        self.inner.rsplitn(n, pred)
    }

    pub fn windows(self, size: uint) -> slice::Windows<'a, T> {
        self.inner.windows(size)
    }

    pub fn chunks(self, size: uint) -> slice::Chunks<'a, T> {
        self.inner.chunks(size)
    }

    pub fn get(&self, index: uint) -> Option<&'a T> {
        self.inner.get(index)
    }

    pub fn head(&self) -> Option<&'a T> {
        self.inner.head()
    }

    pub fn tail(&self) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.tail() }
    }

    pub fn tailn(&self, n: uint) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.tailn(n) }
    }

    pub fn init(&self) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.init() }
    }

    pub fn initn(&self, n: uint) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.initn(n) }
    }

    pub fn last(&self) -> Option<&'a T> {
        self.inner.last()
    }

    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    pub fn bsearch(&self, f: |&T| -> Ordering) -> Option<uint> {
        self.inner.bsearch(f)
    }

    pub fn shift_ref(&mut self) -> Option<&'a T> {
        self.inner.shift_ref()
    }

    pub fn pop_ref(&mut self) -> Option<&'a T> {
        self.inner.pop_ref()
    }
}

impl<'a, T: Ord> /*ImmutableEqVector<'a, T> for*/ SortedSlice<'a, T> {
    pub fn lower_bound(&self, t: &T) -> uint {
        bsearch_no_less(self.inner, |v| v.cmp(t))
    }

    pub fn upper_bound(&self, t: &T) -> uint {
        bsearch_greater(self.inner, |v| v.cmp(t))
    }

    pub fn position_elem(&self, t: &T) -> Option<uint> {
        let i = bsearch_no_less(self.inner, |v| v.cmp(t));
        if i < self.inner.len() && self.inner[i] == *t {
            Some(i)
        } else {
            None
        }
    }

    pub fn rposition_elem(&self, t: &T) -> Option<uint> {
        let i = bsearch_greater(self.inner, |v| v.cmp(t));
        if i > 0 && self.inner[i-1] == *t {
            Some(i-1)
        } else {
            None
        }
    }

    pub fn starts_with<'b>(&self, needle: SortedSlice<'b, T>) -> bool {
        self.inner.starts_with(needle.as_slice())
    }

    pub fn ends_with<'b>(&self, needle: SortedSlice<'b, T>) -> bool {
        self.inner.ends_with(needle.as_slice())
    }
}

impl<'a, T: Ord> /*ImmutableOrdVector<'a, T> for*/ SortedSlice<'a, T> {
    pub fn bsearch_elem(&self, x: &T) -> Option<uint> {
        self.inner.bsearch_elem(x)
    }
}

impl<'a, T: Ord + Clone> CloneableVector<T> for SortedSlice<'a, T> {
    fn to_vec(&self) -> Vec<T> { self.inner.to_vec() }
    fn into_vec(self) -> Vec<T> { self.inner.into_vec() }
}

impl<'a, T: Ord + Clone> /*ImmutableCloneableVector<T> for*/ SortedSlice<'a, T> {
    pub fn partitioned(&self, f: |&T| -> bool) -> (SortedVec<T>, SortedVec<T>) {
        let (yes, no) = self.inner.partitioned(f);
        (SortedVec { inner: yes }, SortedVec { inner: no })
    }

    pub fn permutations(&self) -> slice::Permutations<T> { self.inner.permutations() }
}

impl<'a, T: Ord> Vector<T> for SortedSlice<'a, T> {
    fn as_slice<'a>(&'a self) -> &'a [T] { self.inner.as_slice() }
}

impl<'a, T: Ord> Collection for SortedSlice<'a, T> {
    fn len(&self) -> uint { self.inner.len() }
    fn is_empty(&self) -> bool { self.inner.is_empty() }
}

impl<'a, T: Ord> Default for SortedSlice<'a, T> {
    fn default() -> SortedSlice<'a, T> { SortedSlice { inner: Default::default() } }
}

impl<'a, T: Ord> Set<T> for SortedSlice<'a, T> {
    fn contains(&self, value: &T) -> bool { self.inner.bsearch_elem(value).is_some() }
    fn is_disjoint(&self, other: &SortedSlice<T>) -> bool { is_disjoint(self.inner, other.inner) }
    fn is_subset(&self, other: &SortedSlice<T>) -> bool { is_subset(self.inner, other.inner) }
    fn is_superset(&self, other: &SortedSlice<T>) -> bool { is_subset(other.inner, self.inner) }
}

/// A sorted vector. Equivalent to `Vec<T>` except that it is known to be sorted.
pub struct SortedVec<T> { inner: Vec<T> }

impl<T: Ord> SortedVec<T> {
    pub fn new() -> SortedVec<T> {
        SortedVec { inner: Vec::new() }
    }

    pub fn with_capacity(capacity: uint) -> SortedVec<T> {
        SortedVec { inner: Vec::with_capacity(capacity) }
    }

    pub fn from_fn(length: uint, op: |uint| -> T) -> SortedVec<T> {
        let mut vec = Vec::from_fn(length, op);
        vec.sort();
        SortedVec { inner: vec }
    }

    pub fn from_vec(vec: Vec<T>) -> Result<SortedVec<T>, Vec<T>> {
        if is_sorted(vec.as_slice()) {
            Ok(SortedVec { inner: vec })
        } else {
            Err(vec)
        }
    }

    pub fn from_unsorted_vec(mut vec: Vec<T>) -> SortedVec<T> {
        vec.sort();
        SortedVec { inner: vec }
    }

    pub fn partition(self, f: |&T| -> bool) -> (SortedVec<T>, SortedVec<T>) {
        let (yes, no) = self.inner.partition(f);
        (SortedVec { inner: yes }, SortedVec { inner: no })
    }

    pub fn as_sorted_slice<'a>(&'a self) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.as_slice() }
    }

    pub fn unwrap(self) -> Vec<T> {
        self.inner
    }
}

impl<T: Ord + Clone> SortedVec<T> {
    pub fn append(mut self, second: SortedSlice<T>) -> SortedVec<T> {
        self.push_all(second);
        self
    }

    pub fn from_slice(values: &[T]) -> Option<SortedVec<T>> {
        if is_sorted(values) {
            Some(SortedVec { inner: Vec::from_slice(values) })
        } else {
            None
        }
    }

    pub fn from_unsorted_slice(values: &[T]) -> SortedVec<T> {
        let mut vec = Vec::from_slice(values);
        vec.sort();
        SortedVec { inner: vec }
    }

    pub fn from_sorted_slice(values: SortedSlice<T>) -> SortedVec<T> {
        SortedVec { inner: Vec::from_slice(values.inner) }
    }

    pub fn from_elem(length: uint, value: T) -> SortedVec<T> {
        SortedVec { inner: Vec::from_elem(length, value) }
    }

    pub fn push_all(&mut self, other: SortedSlice<T>) {
        merge_to_sorted_vec(&mut self.inner, other.as_slice());
    }

    pub fn partitioned(&self, f: |&T| -> bool) -> (SortedVec<T>, SortedVec<T>) {
        let (yes, no) = self.inner.partitioned(f);
        (SortedVec { inner: yes }, SortedVec { inner: no })
    }

    pub fn grow(&mut self, n: uint, value: &T) {
        let i = bsearch_greater(self.inner.as_slice(), |v| v.cmp(value));
        insert_many(&mut self.inner, i, n, value);
    }
}

impl<T: Ord> SortedVec<T> {
    pub fn capacity(&self) -> uint {
        self.inner.capacity()
    }

    pub fn reserve_additional(&mut self, extra: uint) {
        self.inner.reserve_additional(extra);
    }

    pub fn reserve(&mut self, capacity: uint) {
        self.inner.reserve(capacity);
    }

    pub fn reserve_exact(&mut self, capacity: uint) {
        self.inner.reserve_exact(capacity);
    }

    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    pub fn append_one(mut self, x: T) -> SortedVec<T> {
        self.push(x);
        self
    }

    pub fn truncate(&mut self, len: uint) {
        self.inner.truncate(len);
    }

    pub unsafe fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] {
        self.inner.as_mut_slice()
    }

    pub fn move_iter(self) -> vec::MoveItems<T> {
        self.inner.move_iter()
    }

    pub fn iter<'a>(&'a self) -> slice::Items<'a, T> {
        self.inner.iter()
    }

    pub unsafe fn mut_iter<'a>(&'a mut self) -> slice::MutItems<'a, T> {
        self.inner.mut_iter()
    }

    pub fn slice<'a>(&'a self, start: uint, end: uint) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.slice(start, end) }
    }

    pub fn tail<'a>(&'a self) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.tail() }
    }

    pub fn tailn<'a>(&'a self, n: uint) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.tailn(n) }
    }

    pub fn last<'a>(&'a self) -> Option<&'a T> {
        self.inner.last()
    }

    pub unsafe fn mut_last<'a>(&'a mut self) -> Option<&'a mut T> {
        self.inner.mut_last()
    }

    pub fn remove(&mut self, index: uint) -> Option<T> {
        self.inner.remove(index)
    }

    pub fn push_all_move(&mut self, other: SortedVec<T>) {
        merge_into_sorted_vec(&mut self.inner, other.inner);
    }

    pub unsafe fn mut_slice<'a>(&'a mut self, start: uint, end: uint) -> &'a mut [T] {
        self.inner.mut_slice(start, end)
    }

    pub unsafe fn mut_slice_to<'a>(&'a mut self, end: uint) -> &'a mut [T] {
        self.inner.mut_slice_to(end)
    }

    pub unsafe fn mut_split_at<'a>(&'a mut self, mid: uint) -> (&'a mut [T], &'a mut [T]) {
        self.inner.mut_split_at(mid)
    }

    pub fn slice_from<'a>(&'a self, start: uint) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.slice_from(start) }
    }

    pub fn slice_to<'a>(&'a self, end: uint) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.slice_to(end) }
    }

    pub fn init<'a>(&'a self) -> SortedSlice<'a, T> {
        SortedSlice { inner: self.inner.init() }
    }

    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }

    pub fn retain(&mut self, f: |&T| -> bool) {
        self.inner.retain(f);
    }

    pub fn grow_fn(&mut self, n: uint, f: |uint| -> T) {
        self.push_all_move(SortedVec::from_fn(n, f));
    }

    pub fn sort(&mut self) {
        // no op
    }

    pub fn contains(&self, x: &T) -> bool {
        self.as_sorted_slice().contains(x)
    }

    pub fn dedup(&mut self) {
        self.inner.dedup();
    }
}

impl<S: Ord + Str> StrVector for SortedVec<S> {
    fn concat(&self) -> String { self.inner.concat() }
    fn connect(&self, sep: &str) -> String { self.inner.connect(sep) }
}

impl<T: Ord + Clone> Clone for SortedVec<T> {
    fn clone(&self) -> SortedVec<T> { SortedVec { inner: self.inner.clone() } }
    fn clone_from(&mut self, other: &SortedVec<T>) { self.inner.clone_from(&other.inner); }
}

impl<T: Ord> Index<uint, T> for SortedVec<T> {
    fn index<'a>(&'a self, index: &uint) -> &'a T { self.inner.index(index) }
}

impl<T: Ord> FromIterator<T> for SortedVec<T> {
    fn from_iter<I: Iterator<T>>(iterator: I) -> SortedVec<T> {
        SortedVec::from_unsorted_vec(FromIterator::from_iter(iterator))
    }
}

impl<T: Ord> Extendable<T> for SortedVec<T> {
    fn extend<I: Iterator<T>>(&mut self, iterator: I) {
        self.push_all_move(FromIterator::from_iter(iterator));
    }
}

impl<T: Ord> PartialEq for SortedVec<T> {
    fn eq(&self, other: &SortedVec<T>) -> bool { self.inner.eq(&other.inner) }
    fn ne(&self, other: &SortedVec<T>) -> bool { self.inner.ne(&other.inner) }
}

impl<T: Ord> PartialOrd for SortedVec<T> {
    fn partial_cmp(&self, other: &SortedVec<T>) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
    fn lt(&self, other: &SortedVec<T>) -> bool { self.inner.lt(&other.inner) }
    fn le(&self, other: &SortedVec<T>) -> bool { self.inner.le(&other.inner) }
    fn gt(&self, other: &SortedVec<T>) -> bool { self.inner.gt(&other.inner) }
    fn ge(&self, other: &SortedVec<T>) -> bool { self.inner.ge(&other.inner) }
}

impl<T: Ord> Eq for SortedVec<T> {}

impl<T: Ord, V: Vector<T>> Equiv<V> for SortedVec<T> {
    fn equiv(&self, other: &V) -> bool { self.inner.equiv(other) }
}

impl<T: Ord> Collection for SortedVec<T> {
    fn len(&self) -> uint { self.inner.len() }
    fn is_empty(&self) -> bool { self.inner.is_empty() }
}

impl<T: Ord + Clone> CloneableVector<T> for SortedVec<T> {
    fn to_vec(&self) -> Vec<T> { self.inner.to_vec() }
    fn into_vec(self) -> Vec<T> { self.inner.into_vec() }
}

impl<T: Ord> Vector<T> for SortedVec<T> {
    fn as_slice<'a>(&'a self) -> &'a [T] { self.inner.as_slice() }
}

impl<T: Ord + Clone, V: Vector<T>> Add<V, SortedVec<T>> for SortedVec<T> {
    fn add(&self, rhs: &V) -> SortedVec<T> {
        let mut lhs = self.clone();
        lhs.push_all(SortedVec::from_unsorted_slice(rhs.as_slice()).as_sorted_slice());
        lhs
    }
}

impl<T: Ord> Default for SortedVec<T> {
    fn default() -> SortedVec<T> { SortedVec { inner: Default::default() } }
}

impl<T: Ord + fmt::Show> fmt::Show for SortedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { self.inner.fmt(f) }
}

impl<T: Ord> Mutable for SortedVec<T> {
    fn clear(&mut self) { self.inner.clear(); }
}

impl<T: Ord> MutableSeq<T> for SortedVec<T> {
    fn push(&mut self, value: T) {
        let i = bsearch_greater(self.inner.as_slice(), |v| v.cmp(&value));
        self.inner.insert(i, value);
    }

    fn pop(&mut self) -> Option<T> { self.inner.pop() }
}

impl<T: Ord> Set<T> for SortedVec<T> {
    fn contains(&self, value: &T) -> bool {
        self.as_sorted_slice().contains(value)
    }
    fn is_disjoint(&self, other: &SortedVec<T>) -> bool {
        self.as_sorted_slice().is_disjoint(&other.as_sorted_slice())
    }
    fn is_subset(&self, other: &SortedVec<T>) -> bool {
        self.as_sorted_slice().is_subset(&other.as_sorted_slice())
    }
    fn is_superset(&self, other: &SortedVec<T>) -> bool {
        self.as_sorted_slice().is_superset(&other.as_sorted_slice())
    }
}

impl<T: Ord> MutableSet<T> for SortedVec<T> {
    fn insert(&mut self, value: T) -> bool {
        let i = bsearch_greater(self.inner.as_slice(), |v| v.cmp(&value));
        if i > 0 && self.inner[i-1] == value {
            false
        } else {
            self.inner.insert(i, value);
            true
        }
    }

    fn remove(&mut self, value: &T) -> bool {
        let i = bsearch_no_less(self.inner.as_slice(), |v| v.cmp(value));
        if i < self.inner.len() && self.inner[i] == *value {
            self.inner.remove(i);
            true
        } else {
            false
        }
    }
}

impl<S: hash::Writer, T: Ord + hash::Hash<S>> hash::Hash<S> for SortedVec<T> {
    fn hash(&self, state: &mut S) { self.inner.hash(state) }
}

