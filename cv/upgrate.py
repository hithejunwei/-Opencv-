#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pip
# pip V10.0.0以上版本需要导入下面的包
from pip._internal.utils.misc import get_installed_distributions
from subprocess import call
from time import sleep

for dist in get_installed_distributions():
    call("pip install --user --upgrade " + dist.project_name, shell=True)

