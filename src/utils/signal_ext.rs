//! Extension traits for Dioxus signals to reduce boilerplate.
//!
//! # Problem
//!
//! Dioxus signals require a verbose pattern for mutations:
//!
//! ```ignore
//! let mut value = signal();  // Read (clones)
//! value.field = new_val;      // Mutate clone
//! signal.set(value);          // Write back
//! ```
//!
//! This pattern appears 20+ times in `components/mod.rs` alone, making code
//! harder to read and maintain.
//!
//! # Solution
//!
//! The `SignalExt` trait provides a `mutate()` method that encapsulates this pattern:
//!
//! ```ignore
//! signal.mutate(|value| {
//!     value.field = new_val;
//! });
//! ```
//!
//! # Examples
//!
//! ## Before: Verbose mutation
//!
//! ```ignore
//! let mut all_batches = batches_signal();
//! all_batches[batch_idx].status = BatchStatus::Processing;
//! all_batches[batch_idx].files[file_idx].progress = 50.0;
//! batches_signal.set(all_batches);
//! ```
//!
//! ## After: Clean mutation
//!
//! ```ignore
//! batches_signal.mutate(|batches| {
//!     batches[batch_idx].status = BatchStatus::Processing;
//!     batches[batch_idx].files[file_idx].progress = 50.0;
//! });
//! ```
//!
//! ## Nested updates
//!
//! ```ignore
//! batches_signal.mutate(|batches| {
//!     for batch in batches.iter_mut() {
//!         if batch.status == BatchStatus::Pending {
//!             batch.status = BatchStatus::Cancelled;
//!         }
//!     }
//! });
//! ```

use dioxus::prelude::*;

/// Extension trait for Dioxus signals providing mutation helpers.
///
/// This trait is automatically implemented for all `Signal<T>` where `T: Clone + 'static`.
pub trait SignalExt<T: Clone + 'static> {
    /// Mutate the signal's value in place.
    ///
    /// This method:
    /// 1. Reads the current value (clones it)
    /// 2. Passes a mutable reference to the closure
    /// 3. Writes the modified value back to the signal
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that mutates the value
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Simple field update
    /// counter_signal.mutate(|count| *count += 1);
    ///
    /// // Complex nested update
    /// state_signal.mutate(|state| {
    ///     state.user.name = "Alice".to_string();
    ///     state.user.age = 30;
    /// });
    ///
    /// // Conditional updates
    /// items_signal.mutate(|items| {
    ///     items.retain(|item| !item.is_deleted);
    /// });
    /// ```
    fn mutate<F>(&mut self, f: F)
    where
        F: FnOnce(&mut T);

    /// Try to mutate the signal's value, rolling back on error.
    ///
    /// This is useful when mutations might fail and you want to leave
    /// the signal unchanged if the closure returns an error.
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that attempts to mutate the value
    ///
    /// # Returns
    ///
    /// `Ok(())` if mutation succeeded, `Err(E)` if closure returned error.
    /// Signal is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Only update if validation passes
    /// state_signal.try_mutate(|state| {
    ///     if new_value.is_valid() {
    ///         state.value = new_value;
    ///         Ok(())
    ///     } else {
    ///         Err("Invalid value")
    ///     }
    /// })?;
    /// ```
    fn try_mutate<F, E>(&mut self, f: F) -> Result<(), E>
    where
        F: FnOnce(&mut T) -> Result<(), E>;

    /// Update the signal by transforming its current value.
    ///
    /// Similar to `mutate`, but for cases where you want to compute
    /// a new value based on the old one.
    ///
    /// # Arguments
    ///
    /// * `f` - Function that transforms the current value into a new value
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Double the counter
    /// counter_signal.update(|count| count * 2);
    ///
    /// // Replace with default if invalid
    /// value_signal.update(|val| {
    ///     if val.is_valid() { val } else { Value::default() }
    /// });
    /// ```
    fn update<F>(&mut self, f: F)
    where
        F: FnOnce(T) -> T;
}

impl<T: Clone + 'static> SignalExt<T> for Signal<T> {
    fn mutate<F>(&mut self, f: F)
    where
        F: FnOnce(&mut T),
    {
        let mut value = self.read().clone();
        f(&mut value);
        self.set(value);
    }

    fn try_mutate<F, E>(&mut self, f: F) -> Result<(), E>
    where
        F: FnOnce(&mut T) -> Result<(), E>,
    {
        let mut value = self.read().clone();
        f(&mut value)?;
        self.set(value);
        Ok(())
    }

    fn update<F>(&mut self, f: F)
    where
        F: FnOnce(T) -> T,
    {
        let old_value = self.read().clone();
        let new_value = f(old_value);
        self.set(new_value);
    }
}

// Note: Tests for SignalExt are omitted because they require a Dioxus runtime.
// The trait methods are simple wrappers around Signal::read() and Signal::set(),
// which are already tested by Dioxus. Integration testing will verify usage
// in actual component contexts.
