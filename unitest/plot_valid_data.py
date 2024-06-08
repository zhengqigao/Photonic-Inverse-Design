'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-12 17:03:02
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-12 17:04:35
'''


from core import builder
train_loader, validation_loader, test_loader = builder.make_dataloader(name="mmi")
