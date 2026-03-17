00000000000036c0 <lb_propagate_step>:
    36c0:	endbr64
    36c4:	mov    0x18(%rdi),%r9
    36c8:	mov    (%rdi),%r10
    36cb:	mov    (%rsi),%rcx
    36ce:	mov    (%rdx),%rdi
    36d1:	imul   %r9,%r9
    36d5:	test   %r9,%r9
    36d8:	je     37d8 <lb_propagate_step+0x118>
    36de:	push   %rbp
    36df:	mov    %r9,%r11
    36e2:	mov    %r9,%rsi
    36e5:	xor    %r8d,%r8d
    36e8:	shl    $0x4,%r11
    36ec:	shr    $1,%rsi
    36ef:	add    %rdi,%r11
    36f2:	shl    $0x5,%rsi
    36f6:	mov    %rsp,%rbp
    36f9:	push   %r12
    36fb:	mov    %r9,%r12
    36fe:	push   %rbx
    36ff:	mov    %r9,%rbx
    3702:	and    $0xfffffffffffffffe,%r12
    3706:	and    $0x1,%ebx
    3709:	nopl   0x0(%rax)
    3710:	cmp    $0x1,%r9
    3714:	je     37d0 <lb_propagate_step+0x110>
    371a:	mov    %r8,%rdx
    371d:	xor    %eax,%eax
    371f:	vxorpd %xmm2,%xmm2,%xmm2
    3723:	shl    $0x4,%rdx
    3727:	add    %r10,%rdx
    372a:	nopw   0x0(%rax,%rax,1)
    3730:	vpermilpd $0xf,(%rdx,%rax,1),%ymm0
    3737:	vpermilpd $0x0,(%rdx,%rax,1),%ymm1
    373e:	vmulpd (%rcx,%rax,1),%ymm0,%ymm0
    3743:	vpermilpd $0x5,(%rcx,%rax,1),%ymm3
    374a:	add    $0x20,%rax
    374e:	cmp    %rsi,%rax
    3751:	vfmsubadd231pd %ymm3,%ymm1,%ymm0
    3756:	vaddpd %ymm0,%ymm2,%ymm2
    375a:	jne    3730 <lb_propagate_step+0x70>
    375c:	vmovapd %xmm2,%xmm0
    3760:	test   %rbx,%rbx
    3763:	vextractf128 $0x1,%ymm2,%xmm2
    3769:	mov    %r12,%rax
    376c:	vaddpd %xmm2,%xmm0,%xmm0
    3770:	je     37a8 <lb_propagate_step+0xe8>
    3772:	lea    (%rax,%r8,1),%rdx
    3776:	shl    $0x4,%rax
    377a:	shl    $0x4,%rdx
    377e:	vmovupd (%rcx,%rax,1),%xmm3
    3783:	vmovupd (%r10,%rdx,1),%xmm1
    3789:	vpermilpd $0x1,%xmm3,%xmm4
    378f:	vpermilpd $0x0,%xmm1,%xmm2
    3795:	vpermilpd $0x3,%xmm1,%xmm1
    379b:	vmulpd %xmm3,%xmm1,%xmm1
    379f:	vfmsubadd231pd %xmm4,%xmm2,%xmm1
    37a4:	vaddpd %xmm1,%xmm0,%xmm0
    37a8:	vpermilpd $0x1,%xmm0,%xmm0
    37ae:	add    $0x10,%rdi
    37b2:	add    %r9,%r8
    37b5:	vmovupd %xmm0,-0x10(%rdi)
    37ba:	cmp    %r11,%rdi
    37bd:	jne    3710 <lb_propagate_step+0x50>
    37c3:	vzeroupper
    37c6:	pop    %rbx
    37c7:	xor    %eax,%eax
    37c9:	pop    %r12
    37cb:	pop    %rbp
    37cc:	ret
    37cd:	nopl   (%rax)
    37d0:	vxorpd %xmm0,%xmm0,%xmm0
    37d4:	xor    %eax,%eax
    37d6:	jmp    3772 <lb_propagate_step+0xb2>
    37d8:	xor    %eax,%eax
    37da:	ret
    37db:	nopl   0x0(%rax,%rax,1)

