text = "爆款/商品/线上/线下/低头族/高头族/网红/网黑/短视频/直播/云游戏/云音乐/网络安全/信息泄露/网络暴力/网络善良/网络诈骗/网络安全/网络游戏/电子竞技/网络小说/网络文学/网络直播/网络电影/网络购物/网络支付/网络教育/在线学习/网络招聘/网络求职/网络社交/网络沉迷/网络科技/网络创新/网络文化/网络传媒/网络医疗/网络医生/网络医院/网络健康/网络犯罪/网络监管/网络管理/网络法律/网络法规/网络安全/抖音/快手/直播/短视频/网红/剁手/囤货/买买买/断货/口罩/消毒液/宅家/居家办公/云办公/云上学/线上教育/直播带货/网课/网购/电商/双十一/双十二/黑五/网抑云/社交距离/新冠/疫苗/核酸检测/健康码/健康宝/熬夜/加班/996/007/云游戏/5G/区块链/人工智能/大数据/物联网/无人驾驶/无人机/机器人/虚拟现实/增强现实/混合现实/元宇宙/密码货币/比特币/以太坊/狗狗币/火箭币/抖币/潮流/潮牌/潮鞋/球鞋/潮服/街头文化/嘻哈/说唱/饭圈/追星/偶像/综艺/真人秀/剧集/电影/电视剧/网剧/动漫/二次元/三次元/宅/腐/耽美/BL/GL/同人/同人志/轻小说/网文/网游/手游/电竞/赛事/直播/主播/UP主/弹幕/评论/吐槽/段子/表情包/热搜/热点/话题/热议/讨论/舆论/公关/危机/炒作/热度/关注/粉丝/转发/点赞/评论/收藏/分享/互动/UGC/PGC/OGC/短视频/直播/播客/音频/视频/图片/文字/内容/创作/创作者/版权/侵权/抄袭/原创/翻译/引用/转载/授权/付费/打赏/广告/推广/营销/推荐/算法/流量/数据/用户/粉丝/关注/互动/社区/平台/APP/网站/公众号/小程序/短链接/二维码/验证码/账号/密码/注册/登录/注销/封号/解封/黑名单/白名单/举报/投诉/反馈/客服/售后/退款/理赔/保修/维修/换货/售前/售中/售后/购物车/订单/支付/发票/物流/配送/签收/评价/好评/差评/中评/晒单/退货/换货/售后/客服/投诉/举报/反馈/建议/意见/满意/不满/期待/希望/感谢/赞美/批评/指责/抱怨/抵制/抗议/声援/支持/鼓励/期待/希望/理解/包容/接受/欣赏/喜欢"
unique_chars = []
for char in text:
    if char not in unique_chars:
        unique_chars.append(char)
print(unique_chars)

unique_chars.remove('/')
# 用/分割输出
print('/'.join(unique_chars))