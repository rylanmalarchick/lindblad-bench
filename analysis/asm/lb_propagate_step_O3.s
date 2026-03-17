0000000000002f90 <lb_propagate_step>:
    2f90:	endbr64
    2f94:	push   %r15
    2f96:	mov    %rdx,%rcx
    2f99:	push   %r14
    2f9b:	push   %r13
    2f9d:	push   %r12
    2f9f:	push   %rbp
    2fa0:	push   %rbx
    2fa1:	sub    $0x38,%rsp
    2fa5:	mov    0x18(%rdi),%r15
    2fa9:	mov    (%rsi),%rax
    2fac:	mov    (%rdi),%rdx
    2faf:	mov    (%rcx),%r13
    2fb2:	imul   %r15,%r15
    2fb6:	test   %r15,%r15
    2fb9:	je     305e <lb_propagate_step+0xce>
    2fbf:	mov    %r15,%r12
    2fc2:	xor    %r14d,%r14d
    2fc5:	shl    $0x4,%r12
    2fc9:	lea    0x0(%r13,%r12,1),%rcx
    2fce:	add    %rax,%r12
    2fd1:	nopl   0x0(%rax)
    2fd8:	mov    %r14,%rbp
    2fdb:	mov    %rax,%rbx
    2fde:	pxor   %xmm5,%xmm5
    2fe2:	shl    $0x4,%rbp
    2fe6:	add    %rdx,%rbp
    2fe9:	nopl   0x0(%rax)
    2ff0:	movupd 0x0(%rbp),%xmm1
    2ff5:	movupd (%rbx),%xmm2
    2ff9:	movapd %xmm1,%xmm4
    2ffd:	movapd %xmm1,%xmm0
    3001:	movapd %xmm2,%xmm3
    3005:	unpckhpd %xmm1,%xmm4
    3009:	shufpd $0x1,%xmm2,%xmm3
    300e:	unpcklpd %xmm1,%xmm0
    3012:	mulpd  %xmm3,%xmm4
    3016:	mulpd  %xmm2,%xmm0
    301a:	movapd %xmm4,%xmm3
    301e:	addpd  %xmm0,%xmm3
    3022:	subpd  %xmm4,%xmm0
    3026:	movapd %xmm3,%xmm4
    302a:	unpckhpd %xmm3,%xmm3
    302e:	ucomisd %xmm0,%xmm3
    3032:	movsd  %xmm0,%xmm4
    3036:	jp     306f <lb_propagate_step+0xdf>
    3038:	add    $0x10,%rbx
    303c:	addpd  %xmm4,%xmm5
    3040:	add    $0x10,%rbp
    3044:	cmp    %rbx,%r12
    3047:	jne    2ff0 <lb_propagate_step+0x60>
    3049:	movups %xmm5,0x0(%r13)
    304e:	add    $0x10,%r13
    3052:	add    %r15,%r14
    3055:	cmp    %rcx,%r13
    3058:	jne    2fd8 <lb_propagate_step+0x48>
    305e:	add    $0x38,%rsp
    3062:	xor    %eax,%eax
    3064:	pop    %rbx
    3065:	pop    %rbp
    3066:	pop    %r12
    3068:	pop    %r13
    306a:	pop    %r14
    306c:	pop    %r15
    306e:	ret
    306f:	movapd %xmm2,%xmm6
    3073:	movapd %xmm1,%xmm0
    3077:	unpckhpd %xmm1,%xmm1
    307b:	mov    %rcx,0x28(%rsp)
    3080:	unpckhpd %xmm6,%xmm6
    3084:	mov    %rdx,0x20(%rsp)
    3089:	add    $0x10,%rbx
    308d:	add    $0x10,%rbp
    3091:	movapd %xmm6,%xmm3
    3095:	mov    %rax,0x8(%rsp)
    309a:	movaps %xmm5,0x10(%rsp)
    309f:	call   49b0 <__muldc3>
    30a4:	movapd 0x10(%rsp),%xmm5
    30aa:	cmp    %rbx,%r12
    30ad:	mov    0x8(%rsp),%rax
    30b2:	unpcklpd %xmm1,%xmm0
    30b6:	mov    0x20(%rsp),%rdx
    30bb:	mov    0x28(%rsp),%rcx
    30c0:	addpd  %xmm0,%xmm5
    30c4:	jne    2ff0 <lb_propagate_step+0x60>
    30ca:	jmp    3049 <lb_propagate_step+0xb9>
    30cf:	nop

