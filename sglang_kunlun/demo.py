# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2026 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了一个hello world的demo类。

Authors: fangzhou07(fangzhou07@baidu.com)
Date:    2026/07/07 22:14:04
"""


class Hello:
    """hello world类"""

    def run(self, name="World", *args):
        """主入口方法。

        根据百度python编码规范，注释应当使用google风格。
        可以使用sphinx配合napoleon扩展插件自动生成文档。

        Args:
            name: 名称

        Returns:
            int类型，执行结果，0表示成功

        Raises:
            ValueError: 参数name的取值不合法
        """
        if not name:
            raise ValueError(name)

        print(f"Hello {name}!")
        return 0
