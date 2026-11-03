"""Contract tests for shared label encoding and saved model configuration."""

import json
import unittest

from scripts.label_config import (
    ASPECT_COLUMNS,
    CLASS_VALUES,
    InvalidLabelError,
    encode_labels,
    get_label_config,
    parse_label,
    validate_label_config,
)


class LabelConfigTests(unittest.TestCase):
    def label_row(self):
        return {
            aspect: str((-2, -1, 0, 1)[index % 4])
            for index, aspect in enumerate(ASPECT_COLUMNS)
        }

    def test_training_and_validation_rows_encode_by_aspect_not_mapping_order(self):
        training = {"id": "train-1", "content": "训练样本", **self.label_row()}
        validation = dict(reversed(list(self.label_row().items())))
        validation.update(id="validation-1", content="验证样本")
        expected = [index % 4 for index in range(len(ASPECT_COLUMNS))]

        self.assertEqual(encode_labels(training), expected)
        self.assertEqual(encode_labels(validation), expected)
        self.assertEqual(training[ASPECT_COLUMNS[0]], "-2")

    def test_parser_accepts_integer_values_and_ascii_literals(self):
        for raw in (-2, -1, 0, 1):
            for value in (raw, str(raw)):
                with self.subTest(value=value):
                    self.assertEqual(parse_label(value), raw)
        self.assertEqual(parse_label("01"), 1)
        self.assertEqual(parse_label("-0"), 0)

    def test_parser_rejects_coercion_missing_and_out_of_domain_values(self):
        cases = [
            ("", "missing"),
            (" \t", "missing"),
            (" 1", "not_integer_literal"),
            ("1 ", "not_integer_literal"),
            ("1\n", "not_integer_literal"),
            ("+1", "not_integer_literal"),
            ("１", "not_integer_literal"),
            ("1.0", "not_integer_literal"),
            ("1e0", "not_integer_literal"),
            ("nan", "not_integer_literal"),
            ("2", "out_of_range"),
            (-3, "out_of_range"),
        ]
        for value, reason in cases:
            with self.subTest(value=value):
                with self.assertRaises(InvalidLabelError) as raised:
                    parse_label(value)
                self.assertEqual(raised.exception.reason, reason)

        for value in (None, False, True, -2.0, -1.2, 0.0, 1.9,
                      float("nan"), float("inf"), [], {}):
            with self.subTest(value=value):
                with self.assertRaises(InvalidLabelError):
                    parse_label(value)

    def test_encoding_rejects_invalid_or_missing_aspect_with_its_name(self):
        aspect = ASPECT_COLUMNS[15]
        for value in (None, False, 1.9, "1.0", " 1", "", "2", "3"):
            row = self.label_row()
            row[aspect] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, aspect):
                    encode_labels(row)

        row = self.label_row()
        del row[aspect]
        with self.assertRaisesRegex(ValueError, aspect):
            encode_labels(row)

    def test_saved_config_survives_json_and_describes_the_actual_encoder(self):
        saved = json.loads(json.dumps(get_label_config()))
        self.assertIsNone(validate_label_config(saved))
        self.assertEqual(saved["aspect_columns"], list(ASPECT_COLUMNS))
        self.assertEqual(saved["class_values"], list(CLASS_VALUES))

        for class_id, raw_value in enumerate(saved["class_values"]):
            row = {aspect: str(raw_value) for aspect in saved["aspect_columns"]}
            self.assertEqual(encode_labels(row), [class_id] * len(ASPECT_COLUMNS))

    def test_loading_rejects_swapped_aspects_or_classes_and_missing_config(self):
        for field in ("aspect_columns", "class_values"):
            saved = get_label_config()
            saved[field][0], saved[field][1] = saved[field][1], saved[field][0]
            with self.subTest(field=field, change="swapped"):
                with self.assertRaises(ValueError):
                    validate_label_config(saved)

            saved = get_label_config()
            del saved[field]
            with self.subTest(field=field, change="missing"):
                with self.assertRaises(ValueError):
                    validate_label_config(saved)

        for saved in (None, {}, []):
            with self.subTest(saved=saved):
                with self.assertRaises(ValueError):
                    validate_label_config(saved)

    def test_loading_rejects_boolean_or_float_class_values(self):
        for value in (False, 0.0):
            saved = get_label_config()
            saved["class_values"][2] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_label_config(saved)

    def test_exported_config_is_independent_of_later_caller_changes(self):
        original = get_label_config()
        edited = get_label_config()
        edited["aspect_columns"].reverse()
        edited["class_values"].clear()

        self.assertEqual(get_label_config(), original)
        self.assertIsNone(validate_label_config(original))


if __name__ == "__main__":
    unittest.main()
