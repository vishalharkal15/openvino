// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/function.hpp"

#include <cstring>

#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convert.hpp"

namespace ov {
namespace reference {

void function(const std::shared_ptr<Model>& function, const ov::TensorVector& inputs, ov::TensorVector& outputs) {
    outputs.reserve(function->get_output_size());
    for (const auto& result : function->get_results()) {
        auto result_type = result->output(0).get_element_type();
        // Workaround for F16/BF16 type compatibility:
        // F16 and BF16 types are not fully supported during constant folding in the reference implementation.
        // When evaluating subgraphs (e.g., in If/Loop operators) that contain F16/BF16 constants,
        // we temporarily use F32 tensors to avoid runtime errors during evaluation.
        // The output tensors will be converted back to the original F16/BF16 type after evaluation.
        if (result_type == element::f16 || result_type == element::bf16) {
            outputs.emplace_back(element::f32, result->output(0).get_shape());
        } else {
            outputs.emplace_back(result->output(0));
        }
    }
    function->evaluate(outputs, inputs);
    
    // Post-processing: Convert F32 tensors back to their original F16/BF16 type
    // This ensures the output maintains the expected element type as specified in the model.
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto expected_type = function->get_results()[i]->output(0).get_element_type();
        if (outputs[i].get_element_type() != expected_type) {
            ov::Tensor converted_output(expected_type, outputs[i].get_shape());
            ov::TensorVector convert_outputs = {converted_output};
            ov::op::v0::Convert().evaluate(convert_outputs, ov::TensorVector{outputs[i]});
            outputs[i] = converted_output;
        }
    }
}

void function(const std::shared_ptr<Model>& function,
              const ov::TensorVector& inputs,
              ov::TensorVector& outputs,
              const EvaluationContext& evaluation_context) {
    outputs.reserve(function->get_output_size());
    for (const auto& result : function->get_results()) {
        auto result_type = result->output(0).get_element_type();
        // Workaround for F16/BF16 type compatibility:
        // F16 and BF16 types are not fully supported during constant folding in the reference implementation.
        // When evaluating subgraphs (e.g., in If/Loop operators) that contain F16/BF16 constants,
        // we temporarily use F32 tensors to avoid runtime errors during evaluation.
        // The output tensors will be converted back to the original F16/BF16 type after evaluation.
        if (result_type == element::f16 || result_type == element::bf16) {
            outputs.emplace_back(element::f32, result->output(0).get_shape());
        } else {
            outputs.emplace_back(result->output(0));
        }
    }
    function->evaluate(outputs, inputs, const_cast<EvaluationContext&>(evaluation_context));
    
    // Post-processing: Convert F32 tensors back to their original F16/BF16 type
    // This ensures the output maintains the expected element type as specified in the model.
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto expected_type = function->get_results()[i]->output(0).get_element_type();
        if (outputs[i].get_element_type() != expected_type) {
            ov::Tensor converted_output(expected_type, outputs[i].get_shape());
            ov::TensorVector convert_outputs = {converted_output};
            ov::op::v0::Convert().evaluate(convert_outputs, ov::TensorVector{outputs[i]});
            outputs[i] = converted_output;
        }
    }
}

}  // namespace reference
}  // namespace ov
