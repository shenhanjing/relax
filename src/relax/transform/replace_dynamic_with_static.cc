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

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const ShapeExprNode* op) final {
    std::map<std::string, uint32_t> shape_value = {{"seq_len", 64}};

    tvm::relay::Shape values =
        op->values.Map([this](const PrimExpr& e) { return this->VisitPrimExpr(e); });

    tvm::relay::Shape static_values;
    for (auto& item : values) {
      if (item->GetTypeKey() == "tir.Var") {
        tvm::tir::Var value = Downcast<tvm::tir::Var>(item);
        std::string value_name = value.get()->name_hint;
        auto it = shape_value.find(value_name);
        if (it != shape_value.end()) {
          int32_t num = it->second;
          auto int_imm = tvm::PrimExpr((int32_t)num);
          static_values.push_back(int_imm);
          continue;
        }
      }
      static_values.push_back(item);
    }

    if (static_values.same_as(op->values)) {
      // If values does not change, struct info won't change.
      return GetRef<Expr>(op);
    } else {
      return ShapeExpr(static_values, op->span);
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
