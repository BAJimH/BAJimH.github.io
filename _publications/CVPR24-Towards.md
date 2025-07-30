---
title: "Towards HDR and HFR video from rolling-mixed-bit spikings"
collection: publications
category: conferences
permalink: /publication/CVPR24-Towards
excerpt: ''
date: 2024-06-17
venue: 'Proceeding. of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)'
paperurl: 'https://openaccess.thecvf.com/content/CVPR2024/papers/Chang_Towards_HDR_and_HFR_Video_from_Rolling-Mixed-Bit_Spikings_CVPR_2024_paper.pdf'
citation: 'Yakun Chang, Yeliduosi Xiaokaiti, Yujia Liu, Bin Fan, Zhaojun Huang, Tiejun Huang, and Boxin Shi. Towards HDR and HFR Video from Rolling-Mixed-Bit Spikings. In Proc. of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.'
---

### Abstract

The spiking cameras offer the benefits of high dynamic range (HDR) high temporal resolution and low data redundancy. However reconstructing HDR videos in high-speed conditions using single-bit spikings presents challenges due to the limited bit depth. Increasing the bit depth of the spikings is advantageous for boosting HDR performance but the readout efficiency will be decreased which is unfavorable for achieving a high frame rate (HFR) video. To address these challenges we propose a readout mechanism to obtain rolling-mixed-bit (RMB) spikings which involves interleaving multi-bit spikings within the single-bit spikings in a rolling manner thereby combining the characteristics of high bit depth and efficient readout. Furthermore we introduce RMB-Net for reconstructing HDR and HFR videos. RMB-Net comprises a cross-bit attention block for fusing mixed-bit spikings and a cross-time attention block for achieving temporal fusion. Extensive experiments conducted on synthetic and real-synthetic data demonstrate the superiority of our method. For instance pure 3-bit spikings result in 3 times of data volume whereas our method achieves comparable performance with less than 2% increase in data volume.
