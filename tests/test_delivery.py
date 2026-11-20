import hashlib
import json
import unittest
from io import BytesIO
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 'f3a14ef2a2003e7c007ed02738bf6568e167ca0d6c8b332c9482ffbde526ff06'

class DeliveryTests(unittest.TestCase):
    def test_model_and_report_identity(self):
        self.assertEqual(hashlib.sha256((ROOT/'models/mealradar_pipeline.joblib').read_bytes()).hexdigest(), EXPECTED)
        report = json.loads((ROOT/'reports/metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(report['pipeline_sha256'], EXPECTED)
        self.assertEqual(report['model_parameters']['class_weight'], 'balanced')
        self.assertEqual(sum(map(sum, report['final_test']['confusion_matrix_true_rows_pred_columns'])), 1193)

    def test_balanced_page_report_and_case(self):
        import joblib
        model = joblib.load(ROOT/'models/mealradar_pipeline.joblib')
        self.assertEqual(model.named_steps['classifier'].class_weight, 'balanced')
        case = json.loads((ROOT/'reports/error_case.json').read_text(encoding='utf-8'))
        self.assertEqual(case['pipeline_sha256'], EXPECTED)
        self.assertEqual(int(model.predict([case['review']])[0]), case['predicted_label'])
        report = json.loads((ROOT/'reports/metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(report['final_test']['confusion_matrix_true_rows_pred_columns'], [[740,56],[64,333]])
        self.assertFalse(report['migration']['test_set_read'])
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('Balanced' in item.value for item in app.subheader))
        self.assertEqual([item.value for item in app.metric], ['89.94%', '0.8862', '92.96%'])
        self.assertFalse(any('暂时不可用' in item.value for item in app.warning))

    def test_single_flow(self):
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run(timeout=30)
        self.assertFalse(app.exception)
        self.check_single_interactions(app)

    def check_single_interactions(self, app):
        app.button(key='example_positive').click().run()
        self.assertTrue(app.text_area[0].value)
        app.button[2].click().run()
        self.assertFalse(app.exception)
        self.assertIn('正面', app.success[0].value)
        app.button(key='example_negative').click().run()
        app.button[2].click().run()
        self.assertIn('负面', app.warning[0].value)
        app.text_area[0].set_value('   ')
        app.button[2].click().run()
        self.assertIn('请先输入', app.warning[0].value)
        self.assertFalse(app.exception)

    def test_batch_flow(self):
        raw = 'review\n很好吃，送餐速度快\n太难吃了，送餐太慢\n" "\n'.encode('utf-8-sig')
        with patch('streamlit.file_uploader', return_value=BytesIO(raw)):
            app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run(timeout=30)
            app.button[3].click().run()
            self.assertFalse(app.exception)
            results = app.session_state['batch_results'][1]
            self.assertEqual(results['预测情感'].tolist(), ['正面', '负面', '未分析'])
            app.checkbox[0].check().run()
            self.assertEqual(len(app.dataframe[0].value), 1)
            self.assertEqual(app.dataframe[0].value.iloc[0]['预测情感'], '负面')

    def test_batch_bad_column(self):
        with patch('streamlit.file_uploader', return_value=BytesIO(b'wrong\ntext\n')):
            app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run(timeout=30)
            app.button[3].click().run()
            self.assertIn('review', app.error[0].value)
            self.assertFalse(app.exception)

    def test_batch_rejects_unquoted_extra_fields(self):
        raw = b'review\nfood good,delivery bad\nfood bad,delivery good\n'
        with patch('streamlit.file_uploader', return_value=BytesIO(raw)):
            app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
            app.button[3].click().run()
            self.assertTrue(app.error, 'Malformed CSV must not silently truncate review text')
            self.assertFalse(app.exception)

    def test_batch_preserves_quoted_comma_and_blank_rows(self):
        raw = 'review\n"很好吃,送得很快"\n\n太难吃了\n'.encode('gb18030')
        with patch('streamlit.file_uploader', return_value=BytesIO(raw)):
            app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
            app.button[3].click().run()
            self.assertFalse(app.exception)
            results = app.session_state['batch_results'][1]
            self.assertEqual(results.review.tolist(), ['很好吃,送得很快', '', '太难吃了'])
            self.assertEqual(results.iloc[1]['预测情感'], '未分析')

    def test_missing_model_is_friendly(self):
        import streamlit as st
        st.cache_resource.clear()
        with patch('joblib.load', side_effect=FileNotFoundError('test missing model')):
            app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run(timeout=30)
            app.text_area[0].set_value('测试评论')
            app.button[2].click().run()
            self.assertIn('模型文件缺失', app.error[0].value)
            self.assertFalse(app.exception)
        st.cache_resource.clear()
        app.button(key='example_positive').click().run()
        self.assertTrue(app.text_area[0].value)
        app.button[2].click().run()
        self.assertFalse(app.exception)
        self.assertIn('正面', app.success[0].value)
        app.button(key='example_negative').click().run()
        app.button[2].click().run()
        self.assertIn('负面', app.warning[0].value)
        app.text_area[0].set_value('   ')
        app.button[2].click().run()
        self.assertIn('请先输入', app.warning[0].value)
        self.assertFalse(app.exception)

if __name__ == '__main__':
    unittest.main()

