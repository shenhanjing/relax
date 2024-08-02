#include <tvm/driver/driver_api.h>
#include <tvm/ir/function.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include <sstream>

namespace tvm {
namespace relax {

class DynamicToStaticReplacer : public ExprMutator {
 public:
  static Function Replace(Function func, IRModule ctx_module) {
    DynamicToStaticReplacer replacer(std::move(ctx_module));
    func = Downcast<Function>(RemoveAllUnused(replacer(func)));
    return func;
  }

 private:
  explicit DynamicToStaticReplacer(IRModule ctx_module) : ExprMutator(ctx_module) {}

  PrimExpr Var2IntImm(const tir::Var& var) {
    static std::map<std::string, int32_t> shape_value = {{"seq_len", 16}};
    std::string value_name = var.get()->name_hint;
    auto it = shape_value.find(value_name);
    if (it == shape_value.end()) {
      std::cout << "[WARNING]: Var \"" << var << "\" is not defined in variable map." << std::endl;
      return var;
    }
    int32_t num = it->second;
    PrimExpr int_imm = tir::make_const(DataType::Int(64), num);
    return int_imm;
  }

  PrimExpr PrimExpr2Static(const PrimExpr& expr) {
    if (expr->IsInstance<tir::VarNode>()) {
      tir::Var var = Downcast<tir::Var>(expr);
      return Var2IntImm(var);
    } else if (expr->IsInstance<tir::IntImmNode>()) {
      return expr;
    } else if (expr->IsInstance<tir::AddNode>()) {
      tir::Add op = Downcast<tir::Add>(expr);
      PrimExpr a = PrimExpr2Static(op->a);
      PrimExpr b = PrimExpr2Static(op->b);
      const IntImmNode* const_a = a.as<IntImmNode>();
      const IntImmNode* const_b = b.as<IntImmNode>();
      if (const_a && const_b) {
        int64_t value = const_a->value + const_b->value;
        PrimExpr int_imm = tir::make_const(DataType::Int(64), value);
        return int_imm;
      }
      return tir::Add(a, b);
    } else if (expr->IsInstance<tir::SubNode>()) {
      tir::Sub op = Downcast<tir::Sub>(expr);
      PrimExpr a = PrimExpr2Static(op->a);
      PrimExpr b = PrimExpr2Static(op->b);
      const IntImmNode* const_a = a.as<IntImmNode>();
      const IntImmNode* const_b = b.as<IntImmNode>();
      if (const_a && const_b) {
        int64_t value = const_a->value - const_b->value;
        PrimExpr int_imm = tir::make_const(DataType::Int(64), value);
        return int_imm;
      }
      return tir::Sub(a, b);
    } else if (expr->IsInstance<tir::MulNode>()) {
      tir::Mul op = Downcast<tir::Mul>(expr);
      PrimExpr a = PrimExpr2Static(op->a);
      PrimExpr b = PrimExpr2Static(op->b);
      const IntImmNode* const_a = a.as<IntImmNode>();
      const IntImmNode* const_b = b.as<IntImmNode>();
      if (const_a && const_b) {
        int64_t value = const_a->value * const_b->value;
        PrimExpr int_imm = tir::make_const(DataType::Int(64), value);
        return int_imm;
      }
      return tir::Mul(a, b);
    } else if (expr->IsInstance<tir::FloorDivNode>()) {
      tir::FloorDiv op = Downcast<tir::FloorDiv>(expr);
      PrimExpr a = PrimExpr2Static(op->a);
      PrimExpr b = PrimExpr2Static(op->b);
      const IntImmNode* const_a = a.as<IntImmNode>();
      const IntImmNode* const_b = b.as<IntImmNode>();
      if (const_a && const_b) {
        int64_t a_value = const_a->value;
        int64_t b_value = const_b->value;
        int value = a_value / b_value;
        if ((a_value % b_value != 0) && ((a_value < 0) != (b_value < 0))) {
          value--;
        }
        PrimExpr int_imm = tir::make_const(DataType::Int(64), value);
        return int_imm;
      }
      return tir::FloorDiv(a, b);
    } else {
      LOG(FATAL) << expr->GetTypeKey() << " is not implemented.";
      throw;
    }
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const ShapeExprNode* op) final {
    relay::Shape values =
        op->values.Map([this](const PrimExpr& e) { return this->VisitPrimExpr(e); });

    relay::Shape static_values;
    for (const PrimExpr& expr : values) {
      static_values.push_back(this->PrimExpr2Static(expr));
    }
    values = static_values;

    if (values.same_as(op->values)) {
      // If values does not change, struct info won't change.
      return GetRef<Expr>(op);
    } else {
      return ShapeExpr(values, op->span);
    }
  }
};

namespace transform {

Pass ReplaceDynamicToStatic() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return DynamicToStaticReplacer::Replace(f, m);
      };
  return CreateFunctionPass(pass_func, 0, "ReplaceDynamicToStatic", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ReplaceDynamicToStatic")
    .set_body_typed(ReplaceDynamicToStatic);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
