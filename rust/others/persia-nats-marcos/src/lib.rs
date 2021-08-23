use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens;
use syn::{Attribute, FnArg, Ident, ImplItem, PatType, Receiver, ReturnType};

#[allow(dead_code)]
struct NatsSubject {
    attrs: Vec<Attribute>,
    ident: Ident,
    arg: PatType,
    receiver: Receiver,
    output: ReturnType,
}

impl NatsSubject {
    fn subject_ident(&self) -> Ident {
        quote::format_ident!("{}", self.ident)
    }

    fn subscribe_subject_ident(&self) -> Ident {
        quote::format_ident!("subscribe_{}", self.ident)
    }

    fn publish_subject_ident(&self) -> Ident {
        quote::format_ident!("publish_{}", self.ident)
    }

    fn req_type(&self) -> TokenStream2 {
        let arg_type = &self.arg.ty;
        quote::quote! {
            #arg_type
        }
    }

    fn resp_type(&self) -> TokenStream2 {
        let output = &self.output;
        let output = match output {
            ReturnType::Default => syn::Type::Verbatim(quote::quote! {()}),
            ReturnType::Type(_, t) => (**t).clone(),
        };
        quote::quote! {
            #output
        }
    }

    fn subscription(&self, stub_type_name: String) -> TokenStream2 {
        let subject_ident = self.subject_ident();
        let subject_string = subject_ident.to_string();
        let subscribe_ident = self.subscribe_subject_ident();
        let req_type = self.req_type();
        let spawn_task = quote::quote! {
            let instance = self.inner.clone();
            let subscription = persia_futures::smol::block_on(nats_client.subscribe(&subject))?;
            persia_futures::tokio::spawn(async move {
                while let Some(msg) = subscription.next().await {
                    let arg: Result<#req_type, _> = tokio::task::block_in_place(|| {
                        ::persia_speedy::Readable::read_from_buffer(&msg.data)
                    });
                    if let Ok(a) = arg {
                        let result = instance.#subject_ident(a).await;
                        let resp = tokio::task::block_in_place(|| {
                            result.write_to_vec().unwrap()
                        });
                        let result = msg.respond(resp).await;
                        if result.is_err() {
                            tracing::error!("failed to respond for {}", #subject_string);
                        }
                    }
                    else {
                        tracing::error!("failed to decode msg from nats of {}", #subject_string);
                    }
                }
            });
        };
        quote::quote! {
            pub fn #subscribe_ident(&self) -> Result<(), NatsError> {
                let nats_client = self.nats_client.clone();
                let subject = nats_client.get_subject(
                    #stub_type_name,
                    #subject_string,
                    Some(nats_client.replica_info.replica_index),
                );

                #spawn_task

                let subject = nats_client.get_subject(
                    #stub_type_name,
                    #subject_string,
                    None,
                );

                #spawn_task

                Ok(())
            }
        }
    }

    fn publishing(&self, stub_type_name: String) -> TokenStream2 {
        let subject_ident = self.subject_ident();
        let subject_string = subject_ident.to_string();
        let publish_ident = self.publish_subject_ident();
        let resp_type = self.resp_type();
        let req_type = self.req_type();

        quote::quote! {
            pub async fn #publish_ident(
                &self,
                message: &#req_type,
                dst_index: Option<usize>,
            ) -> Result<#resp_type, NatsError> {
                let nats_client = self.nats_client.clone();
                let subject = nats_client.get_subject(
                    #stub_type_name,
                    #subject_string,
                    dst_index,
                );

                let msg = tokio::task::block_in_place(|| {
                    message.write_to_vec().unwrap()
                });

                let resp_bytes: Vec<u8>  = self.nats_client.request(&subject, &msg).await?;
                let response = tokio::task::block_in_place(|| {
                    ::persia_speedy::Readable::read_from_buffer(&resp_bytes)
                });
                match response {
                    Ok(resp) => Ok(resp),
                    Err(_) => Err(NatsError::DecodeError),
                }
            }
        }
    }
}

struct NatsStub {
    subjects: Vec<NatsSubject>,
    stub_type_name: String,
}

impl NatsStub {
    fn responder_impl(&self) -> TokenStream2 {
        let stub_ident = quote::format_ident!("{}", self.stub_type_name);
        let responder_ident = quote::format_ident!("{}Responder", self.stub_type_name);
        let subscriptions: Vec<_> = self
            .subjects
            .iter()
            .map(|x| x.subscription(self.stub_type_name.clone()))
            .collect();
        let subscribe_subjects: Vec<_> = self
            .subjects
            .iter()
            .map(|x| {
                let subscribe_ident = x.subscribe_subject_ident();
                quote::quote! {
                    self.#subscribe_ident().expect("failed to subsrcibe");
                }
            })
            .collect();

        quote::quote! {
            pub struct #responder_ident {
                pub inner: #stub_ident,
                pub nats_client: NatsClient
            }

            impl #responder_ident {
                #( #subscriptions )*

                pub fn spawn_subscriptions(&self) -> Result<(), NatsError> {
                    #( #subscribe_subjects )*
                    Ok(())
                }
            }
        }
    }

    fn publisher_impl(&self) -> TokenStream2 {
        let publisher_ident = quote::format_ident!("{}Publisher", self.stub_type_name);
        let publish: Vec<_> = self
            .subjects
            .iter()
            .map(|x| x.publishing(self.stub_type_name.clone()))
            .collect();

        quote::quote! {
            pub struct #publisher_ident {
                pub nats_client: NatsClient
            }

            impl #publisher_ident {
                #( #publish )*
            }
        }
    }
}

#[proc_macro_attribute]
pub fn stub(_attr: TokenStream, tokens: TokenStream) -> TokenStream {
    let item = syn::parse_macro_input!(tokens as syn::ItemImpl);
    let stub_type_name = item.self_ty.clone().into_token_stream().to_string();
    let subjects: Vec<NatsSubject> = item
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Method(m) => Some(m),
            _ => None,
        })
        .map(|m| {
            let mut args = vec![];
            let mut receiver = None;
            for arg in &m.sig.inputs {
                match arg {
                    FnArg::Receiver(captures) => {
                        if captures.mutability.is_some() {
                            panic!("should be &self");
                        } else if receiver.is_some() {
                            panic!("duplicated self");
                        } else {
                            receiver = Some(captures.clone());
                        }
                    }
                    FnArg::Typed(captures) => match *captures.pat {
                        syn::Pat::Ident(_) => args.push(captures.clone()),
                        _ => panic!("patterns aren't allowd in RPC args"),
                    },
                }
            }
            assert_eq!(args.len(), 1, "only support single argument");
            NatsSubject {
                attrs: m.attrs.clone(),
                ident: m.sig.ident.clone(),
                arg: args[0].clone(),
                receiver: receiver.unwrap(),
                output: m.sig.output.clone(),
            }
        })
        .collect();

    let stub = NatsStub {
        subjects,
        stub_type_name,
    };

    let responder_impl = stub.responder_impl();
    let publisher_impl = stub.publisher_impl();

    (quote::quote! {
        #item

        #responder_impl

        #publisher_impl

    })
    .into()
}
