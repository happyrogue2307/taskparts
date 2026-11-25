#pragma once

#include "../perworker.hpp"

/*---------------------------------------------------------------------*/
/* Context switching */

using _context_pointer = char*;

// Plain version, used by swap() and spawn(). It must NOT be marked
// returns_twice: in spawn() the second return lands on a brand new
// fiber stack, and returns_twice makes GCC keep live values in stack
// slots, which would be garbage on that new stack.
extern "C"
void* _taskparts_ctx_save(_context_pointer);

// Same code as _taskparts_ctx_save (an assembler alias defined below),
// but declared returns_twice so the compiler treats call sites like
// setjmp. Used by capture(), whose second return resumes in the same
// frame after that frame has kept running past the first return.
extern "C" __attribute__((returns_twice))
void* _taskparts_ctx_save_rt(_context_pointer);

asm(R"(
.globl _taskparts_ctx_save
        .type _taskparts_ctx_save, @function
        .align 16
_taskparts_ctx_save:
        .cfi_startproc
        movq %rbx, 0(%rdi)
        movq %rbp, 8(%rdi)
        movq %r12, 16(%rdi)
        movq %r13, 24(%rdi)
        movq %r14, 32(%rdi)
        movq %r15, 40(%rdi)
        leaq 8(%rsp), %rdx
        movq %rdx, 48(%rdi)
        movq (%rsp), %rax
        movq %rax, 56(%rdi)
        xorq %rax, %rax
        ret
        .size _taskparts_ctx_save, .-_taskparts_ctx_save
        .cfi_endproc
.globl _taskparts_ctx_save_rt
        .type _taskparts_ctx_save_rt, @function
        .set _taskparts_ctx_save_rt, _taskparts_ctx_save
)");

// if t (%rsi) is nullptr, then have the matching call to _taskparts_ctx_save() return the value 1.
extern "C"
void _taskparts_ctx_restore(_context_pointer ctx, void* t);
asm(R"(
.globl _taskparts_ctx_restore
        .type _taskparts_ctx_restore, @function
        .align 16
_taskparts_ctx_restore:
        .cfi_startproc
        movq 0(%rdi), %rbx
        movq 8(%rdi), %rbp
        movq 16(%rdi), %r12
        movq 24(%rdi), %r13
        movq 32(%rdi), %r14
        movq 40(%rdi), %r15
        test %rsi, %rsi
        mov $01, %rax
        cmove %rax, %rsi
        mov %rsi, %rax
        movq 56(%rdi), %rdx
        movq 48(%rdi), %rsp
        jmpq *%rdx
        .size _taskparts_ctx_restore, .-_taskparts_ctx_restore
        .cfi_endproc
)");

namespace taskparts {
     
class context {  
public:
  
  typedef char context_type[8*8];
  
  using context_pointer = _context_pointer;
  
  template <class X>
  static
  context_pointer addr(X r) {
    return r;
  }
  
  template <class Value>
  static
  void throw_to(context_pointer ctx, Value val) {
    _taskparts_ctx_restore(ctx, (void*)val);
  }
  
  template <class Value>
  static
  void swap(context_pointer ctx1, context_pointer ctx2, Value val2) {
    if (_taskparts_ctx_save(ctx1)) {
      return;
    }
    _taskparts_ctx_restore(ctx2, val2);
  }
  
  template <class Value>
  static
  Value capture(context_pointer ctx) {
    void* r = _taskparts_ctx_save_rt(ctx);
    return (Value)r;
  }
  
  template <class Value>
  static
  char* spawn(context_pointer ctx, Value val) {
    Value target;
    if ((target = (Value)_taskparts_ctx_save(ctx))) {
      target->enter(target);
      assert(false);
    }
    static constexpr
    size_t thread_stack_alignb = 16L;
    // size of the call stack of a fiber that is not running on its
    // parent's stack; override by defining TASKPARTS_FIBER_STACK_SZB
#ifndef TASKPARTS_FIBER_STACK_SZB
#define TASKPARTS_FIBER_STACK_SZB (thread_stack_alignb * (1<<13))
#endif
    static constexpr
    size_t thread_stack_szb = TASKPARTS_FIBER_STACK_SZB;
    static constexpr
    int _X86_64_SP_OFFSET = 6;
    char* stack = (char*)malloc(thread_stack_szb);
    char* sp = &stack[thread_stack_szb];
    sp = (char*)((uintptr_t)sp & -thread_stack_alignb);  // align stack pointer on 16-byte boundary
    sp -= 128; // for red zone
    void** _ctx = (void**)ctx;
    _ctx[_X86_64_SP_OFFSET] = sp;
    return stack;
  }
  
};

class context_wrapper_type {
public:
  context::context_type ctx;
};

perworker::array<context_wrapper_type> ctxs;

context::context_pointer my_ctx() {
  return context::addr(ctxs.mine().ctx);
}

} // end namespace
