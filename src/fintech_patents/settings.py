# coding=utf-8
# Copyright 2020 George Mihaila.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""labels是如何得到的？？？"""
IDS_LABELS = {0: 'insurance',
              1: 'payments',
              2: 'investment',
              3: 'fraud',
              4: 'data analytics',
              5: 'non-fintech'}

LABELS_COLORS = {'insurance': (250, 82, 82),
                 'payments': (243, 72, 197),
                 'investment': (72, 89, 243),
                 'fraud': (72, 243, 226),
                 'data analytics': (157, 243, 72),
                 'non-fintech': (243, 157, 72)}

SAMPLE_ABSTRACT = 'sample_patent_abstract.txt'

# Default config file.
CONFIG_FILE = 'config.ini'
