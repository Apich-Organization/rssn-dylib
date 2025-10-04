//! FFI API for the rssn library.
//!
//! This module provides a C-compatible foreign function interface (FFI) for interacting
//! with the core data structures and functions of the `rssn` library.
//!
//! The primary design pattern used here is the "handle-based" interface. Instead of
//! exposing complex Rust structs directly, which is unsafe and unstable, we expose

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use serde::{Deserialize, Serialize};

use crate::symbolic::core::Expr;

/// A macro to generate the boilerplate for a handle-based FFI API.
///
/// This macro creates three functions for a given type `$T`:
/// - `_from_json`: Deserializes a JSON string into a `$T` object and returns it as a raw pointer (handle).
/// - `_to_json`: Serializes a `$T` object (given by its handle) into a JSON string.
/// - `_free`: Frees the memory of a `$T` object given by its handle.
macro_rules! impl_handle_api {
    ($T:ty, $from_json:ident, $to_json:ident, $free:ident) => {
        #[no_mangle]
        pub extern "C" fn $from_json(json_ptr: *const c_char) -> *mut $T {
            if json_ptr.is_null() {
                return ptr::null_mut();
            }
            let json_str = unsafe { CStr::from_ptr(json_ptr).to_str().unwrap() };
            let result: Result<$T, _> = serde_json::from_str(json_str);
            match result {
                Ok(obj) => Box::into_raw(Box::new(obj)),
                Err(_) => ptr::null_mut(),
            }
        }

        #[no_mangle]
        pub extern "C" fn $to_json(handle: *mut $T) -> *mut c_char {
            if handle.is_null() {
                return ptr::null_mut();
            }
            let obj = unsafe { &*handle };
            let json_str = serde_json::to_string(obj).unwrap();
            CString::new(json_str).unwrap().into_raw()
        }

        #[no_mangle]
        pub extern "C" fn $free(handle: *mut $T) {
            if !handle.is_null() {
                unsafe {
                    let _ = Box::from_raw(handle);
                }
            }
        }
    };
}

// Implement the handle API for `Expr` with the prefix "expr".
impl_handle_api!(Expr, expr_from_json, expr_to_json, expr_free);

/// Returns the string representation of an `Expr` handle.
///
/// The caller is responsible for freeing the returned string using `free_string`.
#[no_mangle]
pub extern "C" fn expr_to_string(handle: *mut Expr) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let expr = unsafe { &*handle };
    let s = CString::new(expr.to_string()).unwrap();
    s.into_raw()
}

use crate::symbolic::simplify::simplify;
use crate::symbolic::unit_unification::unify_expression;

#[derive(Serialize)]
struct FfiResult<T, E> {
    ok: Option<T>,
    err: Option<E>,
}

/// Simplifies an `Expr` and returns a handle to the new, simplified expression.
///
/// The caller is responsible for freeing the returned handle using `expr_free`.
#[no_mangle]
pub extern "C" fn expr_simplify(handle: *mut Expr) -> *mut Expr {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let expr = unsafe { &*handle };
    let simplified_expr = simplify(expr.clone());
    Box::into_raw(Box::new(simplified_expr))
}

/// Attempts to unify the units within an expression.
///
/// Returns a JSON string representing a `FfiResult` which contains either the
/// new `Expr` object in the `ok` field or an error message in the `err` field.
/// The caller can then use `expr_from_json` to get a handle to the new expression.
/// The caller is responsible for freeing the returned string using `free_string`.
#[no_mangle]
pub extern "C" fn expr_unify_expression(handle: *mut Expr) -> *mut c_char {
    if handle.is_null() {
        let result = FfiResult {
            ok: None::<Expr>,
            err: Some("Null pointer passed to expr_unify_expression".to_string()),
        };
        let json_str = serde_json::to_string(&result).unwrap();
        return CString::new(json_str).unwrap().into_raw();
    }
    let expr = unsafe { &*handle };
    let unification_result = unify_expression(expr);

    let ffi_result = match unification_result {
        Ok(unified_expr) => FfiResult {
            ok: Some(unified_expr),
            err: None,
        },
        Err(e) => FfiResult {
            ok: None,
            err: Some(e),
        },
    };

    let json_str = serde_json::to_string(&ffi_result).unwrap();
    CString::new(json_str).unwrap().into_raw()
}

/// Allocates and returns a test string ("pong") to the caller.
///
/// This function serves as a more advanced health check for the FFI interface.
/// It allows the client to verify two things:
/// 1. That the FFI function can be called successfully.
/// 2. That memory allocated in Rust can be safely passed to and then freed by the client
///    by calling `free_string` on the returned pointer.
///
/// Returns a pointer to a null-terminated C string. The caller is responsible for freeing this string.
#[no_mangle]
pub extern "C" fn rssn_test_string_passing() -> *mut c_char {
    let s = CString::new("pong").unwrap();
    s.into_raw()
}

/// Frees a C string that was allocated by this library.
#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}
