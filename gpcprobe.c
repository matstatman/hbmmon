// SPDX-License-Identifier: GPL-2.0
/*
 * gpcprobe - NVIDIA GPU BAR0 register probe for hbmmon.
 *
 * The proprietary nvidia driver owns the GPU BARs exclusively and zeroes the
 * kernel resource entries, so /sys/.../resource0 (EINVAL) and /dev/mem
 * (STRICT_DEVMEM) cannot reach the register window. This module reads the raw
 * BAR0 config register, ioremaps it, and exposes 32-bit reads + one tightly
 * allowlisted write via /dev/gpcprobe (ioctl).
 *
 * GP_IOC_WRITE is hard-allowlisted to exactly the I1500 HBM debug bridge
 * INSTR/MODE control registers across the whole aperture family (broadcast
 * 0x009A0000 + 24 unicast 0x00900000 + i*0x00004000,
 * INSTR/MODE at +0x3CB4/+0x3CB8) — the full write surface of hbmmon's
 * startup DEVICE_ID sweep (arm MODE=0x52 stream + WIR, read the frame
 * back, restore). The DATA latch (+0x3CBC) is deliberately NOT in the
 * list: a DATA-latch write is the MR-write path, and the 2026-09-25
 * canaries showed a WDR DATA-latch write wedges the 1500 bridge
 * card-wide until reboot. The RO shadows (+0x3CC0/4) and STATUS (+0x3CC8)
 * are absent too.
 *
 * Every other offset, on every card, is reject-with-EPERM. The I1500
 * bridge is driven by the PKC-encrypted FB Falcon, so treat those writes
 * as HBM-adjacent: the frontend (gpc_probe.wdr_arm_capture) arms
 * read-only 1500 instructions only, and hbmmon diffs the HBM sentinels
 * (FBPA_NUM_ACTIVE / FBPA_TRAINING) and dmesg Xids after every FBPA,
 * aborting on any change. Since 2026-09-19 the I1500-nhood PLMs are
 * opened at L3 by the SEC2 GSP booter, so host (PL0) writes to these
 * registers land and WIRs are executed by the Falcon (170hx repo
 * docs/hbm-1500-170hx.md §11.7).
 */
#include <linux/module.h>
#include <linux/init.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/pci.h>
#include <linux/pci_regs.h>
#include <linux/uaccess.h>
#include <linux/string.h>

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("hbmmon BAR0 probe (I1500 WDR control writes)");

#define GP_IOC_MAGIC 'G'
struct gp_reg {
	__u32 idx;	/* card index */
	__u32 off;	/* register offset within BAR0 */
	__u32 val;	/* result (out) */
};
struct gp_info {
	__u32 count;
	__u32 bdf[16];	/* (bus << 8) | devfn, 0 = unused */
};
#define GP_IOC_INFO   _IO(GP_IOC_MAGIC, 1)
#define GP_IOC_READ32 _IOWR(GP_IOC_MAGIC, 2, struct gp_reg)
#define GP_IOC_WRITE  _IOW(GP_IOC_MAGIC, 3, struct gp_reg)

/* The only BAR0 offsets GP_IOC_WRITE may touch: the I1500 INSTR/MODE
 * control registers, one block per aperture (broadcast 0x009A0000,
 * unicast 0x00900000 + i*0x00004000, i = 0..23). DATA/STATUS/RO-shadows
 * are intentionally absent (see the file header). */
static int gp_write_allowed(u32 off)
{
	u32 base, offblk;

	if (off >= 0x009A0000 && off < 0x009A4000) {
		base = 0x009A0000;		/* broadcast */
	} else if (off >= 0x00900000 && off < 0x00960000) {
		base = 0x00900000 + (((off - 0x00900000) >> 14) << 14);
	} else {
		return 0;
	}
	offblk = off - base;
	return offblk == 0x3CB4 || offblk == 0x3CB8;
}

#define BAR_ADDR_MASK	0xfffffff0u
#define BAR_MEM_TYPE_IO	0x01u	/* bit 0: 1 = I/O space */
#define BAR_MEM_TYPE_64	0x02u	/* bit 1: 64-bit BAR */

struct gp_card {
	struct pci_dev *pdev;
	volatile void __iomem *map;
	resource_size_t size;
};

static struct gp_card cards[16];
static int ncards;

static long gp_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct gp_reg r;
	struct gp_info info;
	int i;

	switch (cmd) {
	case GP_IOC_INFO:
		memset(&info, 0, sizeof(info));
		info.count = ncards;
		for (i = 0; i < ncards; i++)
			info.bdf[i] = ((u32)cards[i].pdev->bus->number << 8) |
				       cards[i].pdev->devfn;
		return copy_to_user((void __user *)arg, &info, sizeof(info)) ? -EFAULT : 0;
	case GP_IOC_READ32:
		if (copy_from_user(&r, (void __user *)arg, sizeof(r)))
			return -EFAULT;
		if (r.idx >= ncards)
			return -EINVAL;
		if (r.off & 3)
			return -EINVAL;
		if (r.off > cards[r.idx].size - 4)
			return -ERANGE;
		r.val = readl(cards[r.idx].map + r.off);
		if (copy_to_user((void __user *)arg, &r, sizeof(r)))
			return -EFAULT;
		return 0;
		case GP_IOC_WRITE:
		if (copy_from_user(&r, (void __user *)arg, sizeof(r)))
			return -EFAULT;
		if (r.idx >= ncards)
			return -EINVAL;
		if (r.off & 3)
			return -EINVAL;
		if (r.off > cards[r.idx].size - 4)
			return -ERANGE;
		if (!gp_write_allowed(r.off)) {
			pr_warn("gpcprobe: %s write to non-allowlisted 0x%08x refused\n",
				pci_name(cards[r.idx].pdev), r.off);
			return -EPERM;
		}
		pr_info("gpcprobe: %s write 0x%08x <- 0x%08x\n",
			pci_name(cards[r.idx].pdev), r.off, r.val);
		writel(r.val, cards[r.idx].map + r.off);
		return 0;
	}
	return -ENOTTY;
}

static int gp_open(struct inode *inode, struct file *file)
{
	return 0;
}

static const struct file_operations gp_fops = {
	.owner = THIS_MODULE,
	.open = gp_open,
	.unlocked_ioctl = gp_ioctl,
};

static struct miscdevice gp_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "gpcprobe",
	.fops = &gp_fops,
};

static int __init gp_init(void)
{
	struct pci_dev *pdev;
	u32 lo, hi;
	u64 start, len;
	int seen = 0;

	for_each_pci_dev(pdev) {
		seen++;
		if (pdev->vendor == 0x10de || seen <= 12)
			pr_info("gpcprobe: dev %s ven=%04x cls=%06x\n",
				pci_name(pdev), pdev->vendor, pdev->class);
		if (pdev->vendor != 0x10de)
			continue;
		if ((pdev->class >> 16) != 0x03)
			continue;
		if (ncards >= 16)
			break;
		if (pci_read_config_dword(pdev, PCI_BASE_ADDRESS_0, &lo)) {
			pr_err("gpcprobe: %s config read failed\n", pci_name(pdev));
			continue;
		}
		if (lo & BAR_MEM_TYPE_IO) {
			pr_err("gpcprobe: %s BAR0 is I/O space\n", pci_name(pdev));
			continue;
		}
		start = lo & BAR_ADDR_MASK;
		if (lo & BAR_MEM_TYPE_64) {
			if (pci_read_config_dword(pdev, PCI_BASE_ADDRESS_0 + 4, &hi)) {
				pr_err("gpcprobe: %s config read hi failed\n",
					pci_name(pdev));
				continue;
			}
			start |= (u64)hi << 32;
		}
		len = 0x1000000;	/* 16 MiB register window */
		cards[ncards].map = ioremap(start, len);
		if (!cards[ncards].map) {
			pr_err("gpcprobe: %s ioremap 0x%llx failed\n",
				pci_name(pdev), start);
			continue;
		}
		cards[ncards].pdev = pci_dev_get(pdev);
		cards[ncards].size = len;
		ncards++;
		pr_info("gpcprobe: %s BAR0 at 0x%llx (%llu bytes)\n",
			pci_name(pdev), start, len);
	}
	if (!ncards) {
		pr_err("gpcprobe: no NVIDIA GPU BAR0 found (scanned %d devs)\n",
			seen);
		return -ENODEV;
	}
	if (misc_register(&gp_dev))
		goto err;
	return 0;
err:
	while (ncards--) {
		iounmap(cards[ncards].map);
		pci_dev_put(cards[ncards].pdev);
	}
	return -ENODEV;
}

static void __exit gp_exit(void)
{
	misc_deregister(&gp_dev);
	while (ncards--) {
		iounmap(cards[ncards].map);
		pci_dev_put(cards[ncards].pdev);
	}
}

module_init(gp_init);
module_exit(gp_exit);
