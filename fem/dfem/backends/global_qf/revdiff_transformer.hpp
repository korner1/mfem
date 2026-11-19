#pragma once

#include "../local_qf/revdiff_transformer.hpp"
#include "../scratch_bank.hpp"
#include "native_revdiff_transformer.hpp"

namespace mfem::future
{

namespace detail
{

template <typename array_t>
struct WritableQFArray;

template <typename scalar_t, int rank, int... sizes>
struct WritableQFArray<tensor_ndarray<scalar_t, rank, sizes...>>
{
   using type = tensor_ndarray<std::remove_const_t<scalar_t>, rank, sizes...>;
};

template <typename array_t>
static void ZeroQFArray(const array_t array)
{
   auto *data = array.GetData();
   std::size_t num_entries = array.total_size();
   for (const auto size : array_t::tensor_sizes_array()) { num_entries *= size; }
   mfem::forall<UseEnzyme>(num_entries, [=] MFEM_HOST_DEVICE(int entry)
   {
      data[entry] = 0.0_r;
   });
}

template <typename func_t, typename activity_t>
struct GlobalRevDiff : QFWithScratch<real_t>
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

   static_assert(tuple_size<args_tuple>::value == num_inputs + 1,
                 "global functional gradients require exactly one output");
   static_assert(num_active_inputs > 0,
                 "global functional gradients require an active input");
   static_assert(!qfunc_uses_scratch_v<func_t>,
                 "global functional gradients do not yet support QF scratch");

   using output_t = qf_param_decay_t<tuple_element_t<num_inputs, args_tuple>>;
   static_assert(output_t::rank() == 1 && output_t::tensor_rank() == 0,
                 "global functional outputs must be scalar tensor arrays");

   template <std::size_t slot>
   using grad_arg_t = typename WritableQFArray<qf_param_decay_t<
                     tuple_element_t<active_inputs[slot], args_tuple>>>::type;

   template <std::size_t... inputs, std::size_t... slots>
   static FunctionSignature<void(tuple_element_t<inputs, args_tuple>...,
                                 grad_arg_t<slots>&...)>
   MakeSignature(std::index_sequence<inputs...>, std::index_sequence<slots...>);

   using signature = decltype(MakeSignature(
                                std::make_index_sequence<num_inputs>{},
                                std::make_index_sequence<num_active_inputs>{}));

   func_t func;

   GlobalRevDiff(const func_t &function, const int num_points,
                 const std::array<int, num_inputs> &input_sizes) : func(function)
   {
      std::vector<int> sizes{1, 1};
      sizes.insert(sizes.end(), input_sizes.begin(), input_sizes.end());
      SetScratch(std::max(num_points, 1), sizes);
   }

   template <std::size_t... indices>
   MFEM_FORCE_INLINE static void Call(func_t &function,
                    qf_param_decay_t<tuple_element_t<indices, args_tuple>>... args)
   {
      qfunction_wrapper(function, args...);
   }

   template <std::size_t... indices>
   static constexpr auto FunctionPointer(std::index_sequence<indices...>)
   {
      return &Call<indices...>;
   }

   template <std::size_t index>
   static constexpr std::size_t GradientSlot()
   {
      for (std::size_t slot = 0; slot < num_active_inputs; ++slot)
      {
         if (active_inputs[slot] == index) { return slot; }
      }
      return num_active_inputs;
   }

#ifdef MFEM_USE_ENZYME
   template <std::size_t index, typename args_t>
   auto MakeShadow(args_t &args) const
   {
      if constexpr (qf_param_is_active_v<activity_t, index>)
      {
         return get<num_inputs + GradientSlot<index>()>(args);
      }
      else
      {
         using array_t = typename WritableQFArray<qf_param_decay_t<
                         tuple_element_t<index, args_tuple>>>::type;
         array_t shadow(GetScratchPointer(index + 2), get<index>(args).dyn_sizes);
         shadow.strides = get<index>(args).strides;
         ZeroQFArray(shadow);
         return shadow;
      }
   }

   template <typename args_t, std::size_t... indices>
   MFEM_FORCE_INLINE void Reverse(args_t &args, output_t &output, output_t &seed,
                                  std::index_sequence<indices...>) const
   {
      auto shadows = mfem::future::make_tuple(MakeShadow<indices>(args)...);
      __enzyme_autodiff<void>(
         FunctionPointer(std::make_index_sequence<num_inputs + 1>{}),
         enzyme_const, &func, enzyme_dup, &get<indices>(args)..., &output,
         enzyme_interleave, &get<indices>(shadows)..., &seed,
         enzyme_runtime_activity);
   }
#endif

   template <typename... args_t>
   MFEM_FORCE_INLINE void operator()(args_t &...arguments) const
   {
      static_assert(sizeof...(args_t) == num_inputs + num_active_inputs,
                    "expected primal inputs and one gradient per active input");
      tuple<args_t&...> args{arguments...};
      constexpr_for<0, num_active_inputs>([&](auto slot)
      {
         ZeroQFArray(get<num_inputs + slot>(args));
      });
#ifdef MFEM_USE_ENZYME
      output_t output(GetScratchPointer(0), {std::size_t(nq)});
      auto *seed_data = GetScratchPointer(1);
      output_t seed(seed_data, {std::size_t(nq)});
      mfem::forall<UseEnzyme>(nq, [=] MFEM_HOST_DEVICE(int point)
      {
         seed_data[point] = 1.0_r;
      });
      Reverse(args, output, seed, std::make_index_sequence<num_inputs>{});
#else
      static_assert(sizeof...(args_t) == 0,
                    "global functional gradients require Enzyme");
#endif
   }
};

}

template <typename func_t, typename activity_t>
struct create_function_signature<detail::GlobalRevDiff<func_t, activity_t>>
{
   using type = typename detail::GlobalRevDiff<func_t, activity_t>::signature;
};

template <typename activity_t, RevDiffDualMode mode = RevDiffDualMode::Eval,
          typename func_t>
auto make_global_revdiff(
   const func_t &function, const int num_points,
   [[maybe_unused]] const std::array<int, tuple_size<activity_t>::value> &input_sizes)
{
   if constexpr (detail::is_local_qfunc_v<func_t>)
   {
      return make_revdiff<activity_t, mode>(function);
   }
   else
   {
#ifdef MFEM_USE_ENZYME
      return detail::GlobalRevDiff<func_t, activity_t>(function, num_points,
                                                     input_sizes);
#else
      return detail::NativeGlobalRevDiff<func_t, activity_t, mode>(
         function, num_points, input_sizes);
#endif
   }
}

}
