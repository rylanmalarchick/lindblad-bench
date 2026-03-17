0000000000002be0 <lb_propagate_step>:
    2be0:	endbr64
    2be4:	push   %r15
    2be6:	mov    %rdx,%rcx
    2be9:	push   %r14
    2beb:	push   %r13
    2bed:	push   %r12
    2bef:	push   %rbp
    2bf0:	push   %rbx
    2bf1:	sub    $0x38,%rsp
    2bf5:	mov    0x18(%rdi),%r15
    2bf9:	mov    (%rsi),%rax
    2bfc:	mov    (%rdi),%rdx
    2bff:	mov    (%rcx),%r13
    2c02:	imul   %r15,%r15
    2c06:	test   %r15,%r15
    2c09:	je     2cb5 <lb_propagate_step+0xd5>
    2c0f:	mov    %r15,%r12
    2c12:	xor    %r14d,%r14d
    2c15:	shl    $0x4,%r12
    2c19:	lea    0x0(%r13,%r12,1),%rcx
    2c1e:	add    %rax,%r12
    2c21:	nopl   0x0(%rax)
    2c28:	mov    %r14,%rbp
    2c2b:	pxor   %xmm7,%xmm7
    2c2f:	mov    %rax,%rbx
    2c32:	shl    $0x4,%rbp
    2c36:	movapd %xmm7,%xmm6
    2c3a:	add    %rdx,%rbp
    2c3d:	nopl   (%rax)
    2c40:	movsd  0x0(%rbp),%xmm5
    2c45:	movsd  0x8(%rbx),%xmm3
    2c4a:	movsd  0x8(%rbp),%xmm1
    2c4f:	movsd  (%rbx),%xmm2
    2c53:	movapd %xmm5,%xmm4
    2c57:	movapd %xmm3,%xmm0
    2c5b:	mulsd  %xmm1,%xmm0
    2c5f:	movapd %xmm1,%xmm8
    2c64:	mulsd  %xmm2,%xmm4
    2c68:	mulsd  %xmm2,%xmm8
    2c6d:	subsd  %xmm0,%xmm4
    2c71:	movapd %xmm5,%xmm0
    2c75:	mulsd  %xmm3,%xmm0
    2c79:	addsd  %xmm8,%xmm0
    2c7e:	ucomisd %xmm0,%xmm4
    2c82:	jp     2cc6 <lb_propagate_step+0xe6>
    2c84:	add    $0x10,%rbx
    2c88:	addsd  %xmm4,%xmm6
    2c8c:	addsd  %xmm0,%xmm7
    2c90:	add    $0x10,%rbp
    2c94:	cmp    %rbx,%r12
    2c97:	jne    2c40 <lb_propagate_step+0x60>
    2c99:	movsd  %xmm6,0x0(%r13)
    2c9f:	add    $0x10,%r13
    2ca3:	add    %r15,%r14
    2ca6:	movsd  %xmm7,-0x8(%r13)
    2cac:	cmp    %r13,%rcx
    2caf:	jne    2c28 <lb_propagate_step+0x48>
    2cb5:	add    $0x38,%rsp
    2cb9:	xor    %eax,%eax
    2cbb:	pop    %rbx
    2cbc:	pop    %rbp
    2cbd:	pop    %r12
    2cbf:	pop    %r13
    2cc1:	pop    %r14
    2cc3:	pop    %r15
    2cc5:	ret
    2cc6:	movapd %xmm5,%xmm0
    2cca:	mov    %rcx,0x28(%rsp)
    2ccf:	add    $0x10,%rbx
    2cd3:	add    $0x10,%rbp
    2cd7:	mov    %rdx,0x20(%rsp)
    2cdc:	mov    %rax,0x18(%rsp)
    2ce1:	movsd  %xmm7,0x10(%rsp)
    2ce7:	movsd  %xmm6,0x8(%rsp)
    2ced:	call   4580 <__muldc3>
    2cf2:	movsd  0x8(%rsp),%xmm6
    2cf8:	movsd  0x10(%rsp),%xmm7
    2cfe:	cmp    %rbx,%r12
    2d01:	mov    0x18(%rsp),%rax
    2d06:	mov    0x20(%rsp),%rdx
    2d0b:	addsd  %xmm0,%xmm6
    2d0f:	addsd  %xmm1,%xmm7
    2d13:	mov    0x28(%rsp),%rcx
    2d18:	jne    2c40 <lb_propagate_step+0x60>
    2d1e:	jmp    2c99 <lb_propagate_step+0xb9>
    2d23:	data16 cs nopw 0x0(%rax,%rax,1)
    2d2e:	xchg   %ax,%ax

