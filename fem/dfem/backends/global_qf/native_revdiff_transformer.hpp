#pragma once

#include "../local_qf/revdiff_transformer.hpp"

#include <cstring>

namespace mfem::future
{

namespace detail
{

template <typename array_t>
struct NativeGlobalArrayTraits;

template <typename scalar_t, int rank, int... sizes>
struct NativeGlobalArrayTraits<tensor_ndarray<scalar_t, rank, sizes...>>
{
   using scalar_type = std::remove_const_t<scalar_t>;
   using writable_type = tensor_ndarray<scalar_type, rank, sizes...>;
   using shape_type = tensor_ndarray<real_t, rank, sizes...>;
   static constexpr int components = (sizes * ... * 1);
};

template <typename array_t>
MFEM_HOST_DEVICE static std::size_t NativeGlobalPointOffset(
   const array_t &array, const int point, std::size_t component)
{
   std::size_t offset = std::size_t(point) * array.strides[array_t::rank() - 1];
   for (std::size_t axis = 0; axis + 1 < array_t::rank(); ++axis)
   {
      offset += (component % array.size(axis)) * array.strides[axis];
      component /= array.size(axis);
   }
   for (std::size_t axis = 0; axis < array_t::tensor_rank(); ++axis)
   {
      offset += (component % array_t::tensor_size(axis)) *
                array.strides[array_t::rank() + axis];
      component /= array_t::tensor_size(axis);
   }
   return offset;
}

template <typename func_t, typename activity_t, RevDiffDualMode mode>
struct NativeGlobalRevDiff
{
   using primal_signature = typename get_function_signature<func_t>::type;
   using args_tuple = typename primal_signature::parameter_ts;
   static constexpr std::size_t num_inputs = tuple_size<activity_t>::value;
   static constexpr std::size_t num_active_inputs =
      count_active_inputs<activity_t, num_inputs>(
         std::make_index_sequence<num_inputs>{});
   static constexpr auto active_inputs =
      collect_active_inputs<activity_t, num_inputs, num_active_inputs>(
         std::make_index_sequence<num_inputs>{});

   template <std::size_t index>
   using input_t = qf_param_decay_t<tuple_element_t<index, args_tuple>>;

   template <std::size_t slot>
   using grad_arg_t = typename NativeGlobalArrayTraits<
                     input_t<active_inputs[slot]>>::writable_type;

   template <std::size_t... indices, std::size_t... slots>
   static FunctionSignature<void(tuple_element_t<indices, args_tuple>...,
                                 grad_arg_t<slots>&...)>
   MakeSignature(std::index_sequence<indices...>, std::index_sequence<slots...>);

   using signature = decltype(MakeSignature(
                                std::make_index_sequence<num_inputs>{},
                                std::make_index_sequence<num_active_inputs>{}));
   using work_scalar_t = std::conditional_t<mode == RevDiffDualMode::Eval,
         native_dual_t, nested_native_dual_t>;

   NativeGlobalRevDiff(const func_t &function, const int num_points,
                      const std::array<int, num_inputs> &sizes_on_qp)
      : func_(function), num_points_(num_points), sizes_on_qp_(sizes_on_qp)
   {
      MFEM_VERIFY(num_points >= 0, "negative quadrature point count");
      std::vector<int> components;
      for (const int size : sizes_on_qp)
      {
         MFEM_VERIFY(size >= 0, "negative input size per quadrature point");
         components.push_back(std::max(size, 1));
      }
      components.push_back(1);
      scratch_.SetScratch(std::max(num_points, 1), components);
   }

   template <typename... argument_ts>
   void operator()(argument_ts &...arguments) const
   {
      static_assert(std::is_void_v<typename primal_signature::return_t>,
                    "native global functional Q-functions must return void");
      static_assert(tuple_size<args_tuple>::value == num_inputs + 1,
                    "native global functional gradients require one output");
      static_assert(num_active_inputs > 0,
                    "native global functional gradients require an active input");
      static_assert(sizeof...(argument_ts) == num_inputs + num_active_inputs,
                    "expected inputs followed by one gradient per active input");
      static_assert(std::is_same_v<input_t<num_inputs>,
                    tensor_array<native_dual_t>>,
                    "native global functional outputs must be scalar native-dual arrays");
      constexpr_for<0, num_active_inputs>([&](auto slot)
      {
         static_assert(std::is_same_v<typename NativeGlobalArrayTraits<
                       input_t<active_inputs[slot]>>::scalar_type, native_dual_t>,
                       "active native global functional inputs must use native duals");
      });

      if (num_points_ == 0) { return; }
      tuple<argument_ts&...> args{arguments...};
      if constexpr (mode == RevDiffDualMode::Eval)
      {
         func_t function(func_);
         PrepareScratch(function);
         Evaluate(function, args);
      }
      else if constexpr (rebind_qfunc_scalar<func_t,
                         nested_native_dual_t>::supported)
      {
         auto function = MakeNestedFunction<
                         rebind_qfunc_scalar_t<func_t, nested_native_dual_t>>();
         PrepareScratch(function);
         Evaluate(function, args);
      }
      else
      {
         static_assert(rebind_qfunc_scalar<func_t,
                       nested_native_dual_t>::supported,
                       "native global second derivatives require a QFunc<scalar_t, ...> "
                       "type that can be rebound to nested duals");
      }
   }

private:
   func_t func_;
   int num_points_;
   std::array<int, num_inputs> sizes_on_qp_;
   mutable ScratchBank<work_scalar_t> scratch_;

   template <typename working_func_t>
   void PrepareScratch(working_func_t &function) const
   {
      if constexpr (qfunc_uses_scratch_v<working_func_t>)
      {
         function.SetScratch(std::max(num_points_, 1), func_.scratch.components);
         function.SetGlobalScratch(func_.scratch.global);
      }
   }

   template <typename nested_func_t>
   nested_func_t MakeNestedFunction() const
   {
      if constexpr (std::is_constructible_v<nested_func_t, const func_t &>)
      {
         return nested_func_t(func_);
      }
      else if constexpr (std::is_empty_v<func_t>)
      {
         return nested_func_t{};
      }
      else
      {
         static_assert(std::is_trivially_copyable_v<func_t> &&
                       std::is_trivially_copyable_v<nested_func_t> &&
                       sizeof(func_t) == sizeof(nested_func_t) &&
                       alignof(func_t) == alignof(nested_func_t),
                       "native global second derivatives require a converting "
                       "constructor for rebound Q-functions with scalar-dependent "
                       "or nontrivial state");
         nested_func_t function{};
         std::memcpy(static_cast<void *>(&function), &func_, sizeof(func_t));
         return function;
      }
   }

   template <typename array_t, std::size_t index, std::size_t active_input,
             typename source_t>
   array_t MakeInput(const source_t &source, const int seed) const
   {
      using scalar_t = typename NativeGlobalArrayTraits<array_t>::scalar_type;
      static_assert(array_t::rank() == 1 || array_t::rank() == 2,
                    "native global inputs support one or two dynamic dimensions");
      static_assert(std::is_same_v<
                    typename NativeGlobalArrayTraits<array_t>::shape_type,
                    typename NativeGlobalArrayTraits<source_t>::shape_type>,
                    "rebinding a native global Q-function must preserve input shapes");
      static_assert(std::is_same_v<scalar_t, real_t> ||
                    std::is_same_v<scalar_t, work_scalar_t>,
                    "native global inputs must use real or matching dual scalars");
      static_assert(!std::is_same_v<typename NativeGlobalArrayTraits<
                    input_t<index>>::scalar_type, native_dual_t> ||
                    std::is_same_v<scalar_t, work_scalar_t>,
                    "rebinding must preserve every input's derivative slots");
      const int count = num_points_ * sizes_on_qp_[index];
      MFEM_VERIFY(source.size(source_t::rank() - 1) ==
                  std::size_t(num_points_) &&
                  source.total_size() * NativeGlobalArrayTraits<source_t>::components ==
                  std::size_t(count), "native global input size mismatch");
      const auto *source_data = source.GetData();
      auto *data = reinterpret_cast<scalar_t *>(
                      scratch_.GetScratchVector(index).Write());
      mfem::forall(count, [=] MFEM_HOST_DEVICE(int entry)
      {
         scalar_t value{};
         qf_set_flat_value(value, 0, qf_flat_value(source_data[entry], 0));
         if constexpr (mode == RevDiffDualMode::Derivative)
         {
            qf_set_flat_gradient(value, 0, qf_flat_gradient(source_data[entry], 0));
         }
         data[entry] = value;
      });
      if constexpr (index == active_input)
      {
         mfem::forall(num_points_, [=] MFEM_HOST_DEVICE(int point)
         {
            auto &value = data[NativeGlobalPointOffset(source, point, seed)];
            if constexpr (mode == RevDiffDualMode::Eval)
            {
               qf_set_flat_gradient(value, 0, 1.0_r);
            }
            else
            {
               qf_set_flat_value_gradient(value, 0, 1.0_r);
            }
         });
      }
      array_t result(data, source.dyn_sizes);
      result.strides = source.strides;
      return result;
   }

   template <typename working_func_t, std::size_t active_input,
             typename arguments_t, std::size_t... indices>
   auto MakeArguments(arguments_t &args, const int seed,
                      std::index_sequence<indices...>) const
   {
      using working_args = typename get_function_signature<
                           working_func_t>::type::parameter_ts;
      using output_t = qf_param_decay_t<tuple_element_t<num_inputs, working_args>>;
      static_assert(std::is_same_v<output_t, tensor_array<work_scalar_t>>,
                    "rebinding must preserve a scalar output with matching duals");
      auto *output_data = reinterpret_cast<work_scalar_t *>(
                             scratch_.GetScratchVector(num_inputs).Write());
      mfem::forall(num_points_, [=] MFEM_HOST_DEVICE(int point)
      {
         output_data[point] = work_scalar_t{};
      });
      return std::make_tuple(
                MakeInput<qf_param_decay_t<tuple_element_t<indices, working_args>>,
                indices, active_input>(get<indices>(args), seed)...,
                output_t(output_data, {std::size_t(num_points_)}));
   }

   template <typename working_func_t, typename arguments_t>
   void Evaluate(const working_func_t &function, arguments_t &args) const
   {
      constexpr_for<0, num_active_inputs>([&](auto slot)
      {
         constexpr std::size_t input_index = active_inputs[slot];
         const auto &source = get<input_index>(args);
         const auto gradient = get<num_inputs + slot>(args);
         using gradient_t = std::decay_t<decltype(gradient)>;
         static_assert(std::is_same_v<
                       typename NativeGlobalArrayTraits<gradient_t>::shape_type,
                       typename NativeGlobalArrayTraits<input_t<input_index>>::shape_type>,
                       "native global gradients must match their input shapes");
         MFEM_VERIFY(gradient.dyn_sizes == source.dyn_sizes,
                     "native global gradient size mismatch");
         for (int seed = 0; seed < sizes_on_qp_[input_index]; ++seed)
         {
            auto qargs = MakeArguments<working_func_t, input_index>(
                            args, seed, std::make_index_sequence<num_inputs>{});
            if constexpr (qfunc_uses_scratch_v<working_func_t>)
            {
               for (int buffer = 0; buffer < function.scratch.Size(); ++buffer)
               {
                  function.GetScratchVector(buffer) = 0.0_r;
               }
            }
            std::apply([&](auto &...views) { function(views...); }, qargs);

            const auto *output_data = reinterpret_cast<const work_scalar_t *>(
                                         scratch_.GetScratchVector(num_inputs).Read());
            auto *gradient_data = gradient.GetData();
            mfem::forall(num_points_, [=] MFEM_HOST_DEVICE(int point)
            {
               native_dual_t result{};
               if constexpr (mode == RevDiffDualMode::Eval)
               {
                  result.value = qf_flat_gradient(output_data[point], 0);
               }
               else
               {
                  result.value = qf_flat_value_gradient(output_data[point], 0);
                  result.gradient = qf_flat_gradient_gradient(output_data[point], 0);
               }
               gradient_data[NativeGlobalPointOffset(gradient, point, seed)] = result;
            });
         }
      });
   }
};

}

template <typename func_t, typename activity_t, RevDiffDualMode mode>
struct create_function_signature<
      detail::NativeGlobalRevDiff<func_t, activity_t, mode>>
{
   using type = typename detail::NativeGlobalRevDiff<func_t, activity_t,
         mode>::signature;
};

}
