# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Integration tests for the Text Plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import textwrap
import numpy as np
import tensorflow as tf

from tensorboard.backend.event_processing import event_multiplexer
from tensorboard.plugins import base_plugin
from tensorboard.plugins.text import text_plugin

GEMS = ['garnet', 'amethyst', 'pearl', 'steven']


class TextPluginTest(tf.test.TestCase):

  def setUp(self):
    self.logdir = self.get_temp_dir()
    self.generate_testdata()
    multiplexer = event_multiplexer.EventMultiplexer()
    multiplexer.AddRunsFromDirectory(self.logdir)
    multiplexer.Reload()
    context = base_plugin.TBContext(logdir=self.logdir, multiplexer=multiplexer)
    self.plugin = text_plugin.TextPlugin(context)

  def testRoutesProvided(self):
    routes = self.plugin.get_plugin_apps()
    self.assertIsInstance(routes['/tags'], collections.Callable)
    self.assertIsInstance(routes['/text'], collections.Callable)

  def assertConverted(self, actual, expected):
    expected_html = text_plugin.markdown_and_sanitize(expected)
    self.assertEqual(actual, expected_html)

  def generate_testdata(self, include_text=True, logdir=None):
    tf.reset_default_graph()
    sess = tf.Session()
    placeholder = tf.placeholder(tf.string)
    summary_tensor = tf.summary.text('message', placeholder)
    vector_summary = tf.summary.text('vector', placeholder)
    scalar_summary = tf.summary.scalar('twelve', tf.constant(12))

    run_names = ['fry', 'leela']
    for run_name in run_names:
      subdir = os.path.join(logdir or self.logdir, run_name)
      writer = tf.summary.FileWriter(subdir)
      writer.add_graph(sess.graph)

      step = 0
      for gem in GEMS:
        message = run_name + ' *loves* ' + gem
        feed_dict = {
            placeholder: message,
        }
        if include_text:
          summ = sess.run(summary_tensor, feed_dict=feed_dict)
          writer.add_summary(summ, global_step=step)
        step += 1

      vector_message = ['one', 'two', 'three', 'four']
      if include_text:
        summ = sess.run(vector_summary, feed_dict={placeholder: vector_message})
        writer.add_summary(summ)

      summ = sess.run(scalar_summary, feed_dict={placeholder: []})
      writer.add_summary(summ)

      writer.close()

  def testIndex(self):
    index = self.plugin.index_impl()
    self.assertItemsEqual(['fry', 'leela'], index.keys())
    # The summary made via plugin assets (the old method being phased out) is
    # only available for run 'fry'.
    self.assertItemsEqual(['message', 'vector'],
                          index['fry'])
    self.assertItemsEqual(['message', 'vector'], index['leela'])

  def testText(self):
    fry = self.plugin.text_impl('fry', 'message')
    leela = self.plugin.text_impl('leela', 'message')
    self.assertEqual(len(fry), 4)
    self.assertEqual(len(leela), 4)
    for i in range(4):
      self.assertEqual(fry[i]['step'], i)
      self.assertConverted(fry[i]['text'], 'fry *loves* ' + GEMS[i])
      self.assertEqual(leela[i]['step'], i)
      self.assertConverted(leela[i]['text'], 'leela *loves* ' + GEMS[i])

    table = self.plugin.text_impl('fry', 'vector')[0]['text']
    self.assertEqual(table,
                     textwrap.dedent("""\
      <table>
      <tbody>
      <tr>
      <td><p>one</p></td>
      </tr>
      <tr>
      <td><p>two</p></td>
      </tr>
      <tr>
      <td><p>three</p></td>
      </tr>
      <tr>
      <td><p>four</p></td>
      </tr>
      </tbody>
      </table>"""))

  def assertTextConverted(self, actual, expected):
    self.assertEqual(text_plugin.markdown_and_sanitize(actual), expected)

  def testMarkdownConversion(self):
    emphasis = '*Italics1* _Italics2_ **bold1** __bold2__'
    emphasis_converted = ('<p><em>Italics1</em> <em>Italics2</em> '
                          '<strong>bold1</strong> <strong>bold2</strong></p>')

    self.assertEqual(
        text_plugin.markdown_and_sanitize(emphasis), emphasis_converted)

    md_list = textwrap.dedent("""\
    1. List item one.
    2. List item two.
      * Sublist
      * Sublist2
    1. List continues.
    """)
    md_list_converted = textwrap.dedent("""\
    <ol>
    <li>List item one.</li>
    <li>List item two.</li>
    <li>Sublist</li>
    <li>Sublist2</li>
    <li>List continues.</li>
    </ol>""")
    self.assertEqual(
        text_plugin.markdown_and_sanitize(md_list), md_list_converted)

    link = '[TensorFlow](http://tensorflow.org)'
    link_converted = '<p><a href="http://tensorflow.org">TensorFlow</a></p>'
    self.assertEqual(text_plugin.markdown_and_sanitize(link), link_converted)

    table = textwrap.dedent("""\
    An | Example | Table
    --- | --- | ---
    A | B | C
    1 | 2 | 3
    """)

    table_converted = textwrap.dedent("""\
    <table>
    <thead>
    <tr>
    <th>An</th>
    <th>Example</th>
    <th>Table</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>A</td>
    <td>B</td>
    <td>C</td>
    </tr>
    <tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    </tr>
    </tbody>
    </table>""")

    self.assertEqual(text_plugin.markdown_and_sanitize(table), table_converted)

  def testSanitization(self):
    dangerous = "<script>alert('xss')</script>"
    sanitized = "&lt;script&gt;alert('xss')&lt;/script&gt;"
    self.assertEqual(text_plugin.markdown_and_sanitize(dangerous), sanitized)

    dangerous = textwrap.dedent("""\
    hello <a name='n'
    href='javascript:alert('xss')'>*you*</a>""")
    sanitized = '<p>hello <a><em>you</em></a></p>'
    self.assertEqual(text_plugin.markdown_and_sanitize(dangerous), sanitized)

  def testTableGeneration(self):
    array2d = np.array([['one', 'two'], ['three', 'four']])
    expected_table = textwrap.dedent("""\
    <table>
    <tbody>
    <tr>
    <td>one</td>
    <td>two</td>
    </tr>
    <tr>
    <td>three</td>
    <td>four</td>
    </tr>
    </tbody>
    </table>""")
    self.assertEqual(text_plugin.make_table(array2d), expected_table)

    expected_table_with_headers = textwrap.dedent("""\
    <table>
    <thead>
    <tr>
    <th>c1</th>
    <th>c2</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>one</td>
    <td>two</td>
    </tr>
    <tr>
    <td>three</td>
    <td>four</td>
    </tr>
    </tbody>
    </table>""")

    actual_with_headers = text_plugin.make_table(array2d, headers=['c1', 'c2'])
    self.assertEqual(actual_with_headers, expected_table_with_headers)

    array_1d = np.array(['one', 'two', 'three', 'four', 'five'])
    expected_1d = textwrap.dedent("""\
    <table>
    <tbody>
    <tr>
    <td>one</td>
    </tr>
    <tr>
    <td>two</td>
    </tr>
    <tr>
    <td>three</td>
    </tr>
    <tr>
    <td>four</td>
    </tr>
    <tr>
    <td>five</td>
    </tr>
    </tbody>
    </table>""")
    self.assertEqual(text_plugin.make_table(array_1d), expected_1d)

    expected_1d_with_headers = textwrap.dedent("""\
    <table>
    <thead>
    <tr>
    <th>X</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>one</td>
    </tr>
    <tr>
    <td>two</td>
    </tr>
    <tr>
    <td>three</td>
    </tr>
    <tr>
    <td>four</td>
    </tr>
    <tr>
    <td>five</td>
    </tr>
    </tbody>
    </table>""")
    actual_1d_with_headers = text_plugin.make_table(array_1d, headers=['X'])
    self.assertEqual(actual_1d_with_headers, expected_1d_with_headers)

  def testMakeTableExceptions(self):
    # Verify that contents is being type-checked and shape-checked.
    with self.assertRaises(ValueError):
      text_plugin.make_table([])

    with self.assertRaises(ValueError):
      text_plugin.make_table('foo')

    with self.assertRaises(ValueError):
      invalid_shape = np.full((3, 3, 3), 'nope', dtype=np.dtype('S3'))
      text_plugin.make_table(invalid_shape)

    # Test headers exceptions in 2d array case.
    test_array = np.full((3, 3), 'foo', dtype=np.dtype('S3'))
    with self.assertRaises(ValueError):
      # Headers is wrong type.
      text_plugin.make_table(test_array, headers='foo')
    with self.assertRaises(ValueError):
      # Too many headers.
      text_plugin.make_table(test_array, headers=['foo', 'bar', 'zod', 'zoink'])
    with self.assertRaises(ValueError):
      # headers is 2d
      text_plugin.make_table(test_array, headers=test_array)

    # Also make sure the column counting logic works in the 1d array case.
    test_array = np.array(['foo', 'bar', 'zod'])
    with self.assertRaises(ValueError):
      # Too many headers.
      text_plugin.make_table(test_array, headers=test_array)

  def test_reduce_to_2d(self):

    def make_range_array(dim):
      """Produce an incrementally increasing multidimensional array.

      Args:
        dim: the number of dimensions for the array

      Returns:
        An array of increasing integer elements, with dim dimensions and size
        two in each dimension.

      Example: rangeArray(2) results in [[0,1],[2,3]].
      """
      return np.array(range(2**dim)).reshape([2] * dim)

    for i in range(2, 5):
      actual = text_plugin.reduce_to_2d(make_range_array(i))
      expected = make_range_array(2)
      np.testing.assert_array_equal(actual, expected)

  def test_text_array_to_html(self):

    convert = text_plugin.text_array_to_html
    scalar = np.array('foo')
    scalar_expected = '<p>foo</p>'
    self.assertEqual(convert(scalar), scalar_expected)

    table = textwrap.dedent("""\
    An | Example | Table
    --- | --- | ---
    A | B | C
    1 | 2 | 3
    """)

    table_converted = textwrap.dedent("""\
    <table>
    <thead>
    <tr>
    <th>An</th>
    <th>Example</th>
    <th>Table</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>A</td>
    <td>B</td>
    <td>C</td>
    </tr>
    <tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    </tr>
    </tbody>
    </table>""")
    
    scalar = np.array(table)
    scalar_expected = table_converted
    self.assertEqual(convert(scalar), scalar_expected)

    vector = np.array(['foo', 'bar'])
    vector_expected = textwrap.dedent("""\
      <table>
      <tbody>
      <tr>
      <td><p>foo</p></td>
      </tr>
      <tr>
      <td><p>bar</p></td>
      </tr>
      </tbody>
      </table>""")
    self.assertEqual(convert(vector), vector_expected)

    d2 = np.array([['foo', 'bar'], ['zoink', 'zod']])
    d2_expected = textwrap.dedent("""\
      <table>
      <tbody>
      <tr>
      <td><p>foo</p></td>
      <td><p>bar</p></td>
      </tr>
      <tr>
      <td><p>zoink</p></td>
      <td><p>zod</p></td>
      </tr>
      </tbody>
      </table>""")
    self.assertEqual(convert(d2), d2_expected)

    d3 = np.array([[['foo', 'bar'], ['zoink', 'zod']], [['FOO', 'BAR'],
                                                        ['ZOINK', 'ZOD']]])

    warning = text_plugin.markdown_and_sanitize(text_plugin.WARNING_TEMPLATE %
                                                3)
    d3_expected = warning + textwrap.dedent("""\
      <table>
      <tbody>
      <tr>
      <td><p>foo</p></td>
      <td><p>bar</p></td>
      </tr>
      <tr>
      <td><p>zoink</p></td>
      <td><p>zod</p></td>
      </tr>
      </tbody>
      </table>""")
    self.assertEqual(convert(d3), d3_expected)
  
  def testPluginIsActiveWhenNoRuns(self):
    """The plugin should be inactive when there are no runs."""
    multiplexer = event_multiplexer.EventMultiplexer()
    context = base_plugin.TBContext(logdir=None, multiplexer=multiplexer)
    plugin = text_plugin.TextPlugin(context)
    self.assertFalse(plugin.is_active())

  def testPluginIsActiveWhenTextRuns(self):
    """The plugin should be active when there are runs with text."""
    multiplexer = event_multiplexer.EventMultiplexer()
    context = base_plugin.TBContext(logdir=None, multiplexer=multiplexer)
    plugin = text_plugin.TextPlugin(context)
    multiplexer.AddRunsFromDirectory(self.logdir)
    multiplexer.Reload()
    self.assertTrue(plugin.is_active())

  def testPluginIsActiveWhenRunsButNoText(self):
    """The plugin should be inactive when there are runs but none has text."""
    multiplexer = event_multiplexer.EventMultiplexer()
    context = base_plugin.TBContext(logdir=None, multiplexer=multiplexer)
    plugin = text_plugin.TextPlugin(context)
    logdir = os.path.join(self.get_temp_dir(), 'runs_with_no_text')
    self.generate_testdata(include_text=False, logdir=logdir)
    multiplexer.AddRunsFromDirectory(logdir)
    multiplexer.Reload()
    self.assertFalse(plugin.is_active())

  def testUnicode(self):
    self.assertConverted(u'<p>Iñtërnâtiônàlizætiøn⚡💩</p>',
                         'Iñtërnâtiônàlizætiøn⚡💩')


class TextPluginBackwardsCompatibilityTest(tf.test.TestCase):

  def setUp(self):
    self.logdir = self.get_temp_dir()
    self.generate_testdata()
    multiplexer = event_multiplexer.EventMultiplexer()
    multiplexer.AddRunsFromDirectory(self.logdir)
    multiplexer.Reload()
    context = base_plugin.TBContext(logdir=self.logdir, multiplexer=multiplexer)
    self.plugin = text_plugin.TextPlugin(context)

  def generate_testdata(self):
    tf.reset_default_graph()
    sess = tf.Session()
    placeholder = tf.constant('I am deprecated.')

    # Previously, we had used a means of creating text summaries that used
    # plugin assets (which loaded JSON files containing runs and tags). The
    # plugin must continue to be able to load summaries of that format, so we
    # create a summary using that old plugin asset-based method here.
    plugin_asset_summary = tf.summary.tensor_summary('old_plugin_asset_summary',
                                                     placeholder)
    assets_directory = os.path.join(self.logdir, 'fry', 'plugins',
                                    'tensorboard_text')
    # Make the directory of assets if it does not exist.
    if not os.path.isdir(assets_directory):
      try:
        os.makedirs(assets_directory)
      except OSError as err:
        self.assertFail('Could not make assets directory %r: %r',
                        assets_directory, err)
    json_path = os.path.join(assets_directory, 'tensors.json')
    with open(json_path, 'w+') as tensors_json_file:
      # Write the op name to a JSON file that the text plugin later uses to
      # determine the tag names of tensors to fetch.
      tensors_json_file.write(json.dumps([plugin_asset_summary.op.name]))

    run_name = 'fry'
    subdir = os.path.join(self.logdir, run_name)
    writer = tf.summary.FileWriter(subdir)
    writer.add_graph(sess.graph)

    summ = sess.run(plugin_asset_summary)
    writer.add_summary(summ)
    writer.close()

  def testIndex(self):
    index = self.plugin.index_impl()
    self.assertItemsEqual(['fry'], index.keys())
    # The summary made via plugin assets (the old method being phased out) is
    # only available for run 'fry'.
    self.assertItemsEqual(['old_plugin_asset_summary'],
                          index['fry'])

  def testText(self):
    fry = self.plugin.text_impl('fry', 'old_plugin_asset_summary')
    self.assertEqual(len(fry), 1)
    self.assertEqual(fry[0]['step'], 0)
    self.assertEqual(fry[0]['text'], u'<p>I am deprecated.</p>')



if __name__ == '__main__':
  tf.test.main()
