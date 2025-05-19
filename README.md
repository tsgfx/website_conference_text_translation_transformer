# 基于Transformer的中英文会议文本智能翻译模型设计与实现
copyright@Guo ruichao 

email: grcsxb269@163.com

## 目录结构及配置介绍

- checkpoint目录为存储模型训练权重的目录，需要将权重文件加入该目录即可使用训练好的模型。如果自行训练，则会在该目录中自动生成权重文件。下面是训练好的中译英和英译中模型的权重下载链接：

  英译中模型权重：[model_bpe_50000_large_epoch40_not_share]( https://pan.baidu.com/s/1kBc4mF841_PdPDPhvMYUwQ?pwd=mjam)

  中译英模型权重：[model_bpe_1000000_large_epoch40_zh2en_not_share]( https://pan.baidu.com/s/1kBc4mF841_PdPDPhvMYUwQ?pwd=mjam)

- [train_data_size_1000000]( https://pan.baidu.com/s/1kBc4mF841_PdPDPhvMYUwQ?pwd=mjam)目录中存放着1000000行用于英译中的中英对照数据集的数据集和词表信息，点击前面的链接即可下载。
  --来自百度网盘超级会员v4的分享

- [train_data_size_1000000]( https://pan.baidu.com/s/1kBc4mF841_PdPDPhvMYUwQ?pwd=mjam)目录中存放着1000000行用于中译英的中英对照数据集的数据集和词表信息，点击前面的链接即可下载。

- translation和website_conference_text_translation_transformer目录为实现翻译系统的前后端实现。

- transformer_bpe_50000_large.ipynb和transformer_bpe_50000_large_zh2en.ipynb为模型的训练文件

- transformer_bpe_50000_large.py和transformer_bpe_50000_large_zh2en.py为后端需要调用transformer的模块实现。

- 数据集来自联合国公开数据集，也可自行下载：

  [UNv1.0.en-zh.tar.gz.00](https://www.un.org/dgacm/sites/www.un.org.dgacm/files/files/UNCORPUS/UNv1.0.en-zh.tar.gz.00)

  [UNv1.0.en-zh.tar.gz.01](https://www.un.org/dgacm/sites/www.un.org.dgacm/files/files/UNCORPUS/UNv1.0.en-zh.tar.gz.01)

  

## Tensorboard可视化
`python -m tensorboard.main --logdir=G:\contrast_and_ablation_experiments\contrast_and_ablation_experiments\runs --host localhost --port 8848`

## 测试数据
1. The committee welcomes all participants to this important session focused on international cooperation in space technology and sustainable development. Over the past years, our working group has made significant progress in promoting peaceful uses of outer space, and today’s discussion will build on those achievements. We appreciate the contributions of all member states and organizations in advancing this agenda.
委员会欢迎所有与会者参加这次重点讨论空间技术和可持续发展方面的国际合作的重要会议。过去几年，我们的工作组在促进和平利用外层空间方面取得了重大进展，今天的讨论将在这些成就的基础上进行。我们赞赏所有会员国和组织为推进这一议程所作的贡献。
2. Despite our efforts, several challenges remain, including the growing issue of space debris, the need for equitable access to satellite data, and financial constraints in developing countries. The report before us highlights these obstacles and proposes measures to address them. We must ensure that technical assistance and capacity-building remain central to our resolution.
尽管我们作出了努力，但仍然存在一些挑战，包括日益严重的空间碎片问题、公平获取卫星数据的必要性以及发展中国家的财政限制。我们面前的报告突出了这些障碍，并提出了解决这些障碍的措施。我们必须确保技术援助和能力建设仍然是我们决议的核心。
3. The draft resolution suggests strengthening regional collaboration, enhancing remote sensing applications, and establishing a fund to support research and innovation. Additionally, the Secretariat has recommended training programs to improve data analysis skills. We urge all delegations to consider these proposals and provide feedback during today’s debate.
该决议草案建议加强区域合作，加强遥感应用，并建立支持研究和创新的基金。此外，秘书处还建议了提高数据分析技能的培训方案。我们敦促所有代表团考虑这些建议，并在今天的辩论中提供反馈。
4. Following this meeting, the subcommittee will finalize its recommendations for submission to the General Assembly. We encourage governments and private sector partners to increase their support for these initiatives. A follow-up conference is planned for 1997 to review implementation and assess impact.
在这次会议之后，小组委员会将最后确定其提交大会的建议。我们鼓励各国政府和私营部门合作伙伴加大对这些倡议的支持。计划在1997年举行一次后续会议，审查执行情况和评价影响。
5. In conclusion, we thank all participants for their commitment to this mission. By working together, we can ensure that space technology benefits all humanity, in line with the principles of the United Nations. Let us move forward with determination and unity.
最后，我们感谢所有与会者对这一使命的承诺。通过共同努力，我们可以确保空间技术按照联合国的原则造福全人类。让我们坚定团结地前进。
6. At the recent international conference held under the auspices of the United Nations, representatives from over fifty countries gathered to discuss ongoing global security concerns. The Secretary-General emphasized the importance of international cooperation and reaffirmed the collective mandate to support regions in crisis. Delegates reviewed recent resolutions adopted by the Security Council, highlighting the need for strengthened missions and increased international assistance. Strongly condemning the persistent threats to civilian safety worldwide, the participants urged for a renewed commitment to peacekeeping efforts. This meeting reflected the international community’s unified stance on promoting stability and reinforcing cooperation among member states.
在最近由联合国主持召开的国际会议上，来自50多个国家的代表齐聚一堂，讨论当前的全球安全问题。秘书长强调了国际合作的重要性，并重申了支持危机区域的集体任务。代表们审查了安全理事会最近通过的决议，强调需要加强特派团和增加国际援助。与会者强烈谴责对全世界平民安全的持续威胁，敦促重新致力于维持和平努力。这次会议反映了国际社会在维护稳定、加强成员国合作方面的统一立场。