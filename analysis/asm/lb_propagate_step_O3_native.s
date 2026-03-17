00000000000036e0 <lb_propagate_step>:
    36e0:	endbr64
    36e4:	push   %r15
    36e6:	mov    %rdx,%rcx
    36e9:	push   %r14
    36eb:	push   %r13
    36ed:	push   %r12
    36ef:	push   %rbp
    36f0:	push   %rbx
    36f1:	sub    $0x38,%rsp
    36f5:	mov    0x18(%rdi),%r15
    36f9:	mov    (%rsi),%rax
    36fc:	mov    (%rdi),%rdx
    36ff:	mov    (%rcx),%r13
    3702:	imul   %r15,%r15
    3706:	test   %r15,%r15
    3709:	je     3798 <lb_propagate_step+0xb8>
    370f:	mov    %r15,%r12
    3712:	xor    %r14d,%r14d
    3715:	shl    $0x4,%r12
    3719:	lea    0x0(%r13,%r12,1),%rcx
    371e:	add    %rax,%r12
    3721:	nopl   0x0(%rax)
    3728:	mov    %r14,%rbp
    372b:	vxorpd %xmm6,%xmm6,%xmm6
    372f:	mov    %rax,%rbx
    3732:	shl    $0x4,%rbp
    3736:	vmovsd %xmm6,%xmm6,%xmm7
    373a:	add    %rdx,%rbp
    373d:	nopl   (%rax)
    3740:	vmovsd 0x8(%rbp),%xmm1
    3745:	vmovsd (%rbx),%xmm2
    3749:	vmovsd 0x8(%rbx),%xmm3
    374e:	vmovsd 0x0(%rbp),%xmm5
    3753:	vmulsd %xmm2,%xmm1,%xmm0
    3757:	vmulsd %xmm3,%xmm1,%xmm4
    375b:	vfmadd231sd %xmm3,%xmm5,%xmm0
    3760:	vfmsub231sd %xmm2,%xmm5,%xmm4
    3765:	vucomisd %xmm0,%xmm4
    3769:	jp     37a9 <lb_propagate_step+0xc9>
    376b:	add    $0x10,%rbx
    376f:	vaddsd %xmm4,%xmm7,%xmm7
    3773:	vaddsd %xmm0,%xmm6,%xmm6
    3777:	add    $0x10,%rbp
    377b:	cmp    %rbx,%r12
    377e:	jne    3740 <lb_propagate_step+0x60>
    3780:	vmovsd %xmm7,0x0(%r13)
    3786:	add    $0x10,%r13
    378a:	add    %r15,%r14
    378d:	vmovsd %xmm6,-0x8(%r13)
    3793:	cmp    %r13,%rcx
    3796:	jne    3728 <lb_propagate_step+0x48>
    3798:	add    $0x38,%rsp
    379c:	xor    %eax,%eax
    379e:	pop    %rbx
    379f:	pop    %rbp
    37a0:	pop    %r12
    37a2:	pop    %r13
    37a4:	pop    %r14
    37a6:	pop    %r15
    37a8:	ret
    37a9:	vmovsd %xmm5,%xmm5,%xmm0
    37ad:	mov    %rcx,0x28(%rsp)
    37b2:	add    $0x10,%rbx
    37b6:	add    $0x10,%rbp
    37ba:	mov    %rdx,0x18(%rsp)
    37bf:	mov    %rax,0x10(%rsp)
    37c4:	vmovsd %xmm6,0x20(%rsp)
    37ca:	vmovsd %xmm7,0x8(%rsp)
    37d0:	call   5b30 <__muldc3>
    37d5:	vmovsd 0x8(%rsp),%xmm7
    37db:	vmovsd 0x20(%rsp),%xmm6
    37e1:	cmp    %rbx,%r12
    37e4:	mov    0x10(%rsp),%rax
    37e9:	mov    0x18(%rsp),%rdx
    37ee:	vaddsd %xmm0,%xmm7,%xmm7
    37f2:	vaddsd %xmm1,%xmm6,%xmm6
    37f6:	mov    0x28(%rsp),%rcx
    37fb:	jne    3740 <lb_propagate_step+0x60>
    3801:	jmp    3780 <lb_propagate_step+0xa0>
    3806:	cs nopw 0x0(%rax,%rax,1)

